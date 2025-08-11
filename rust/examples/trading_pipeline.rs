//! Example: Full Trading Pipeline with Grad-CAM
//!
//! This example demonstrates a complete trading pipeline that:
//! 1. Fetches real-time data from Bybit
//! 2. Generates predictions using a CNN
//! 3. Explains predictions using Grad-CAM
//! 4. Makes trading decisions based on confidence and interpretability
//!
//! Run with: cargo run --example trading_pipeline

use gradcam_finance::{
    BybitClient, CNNConfig, DataProcessor, FeatureEngineering, FinancialCNN, GradCAM,
    GradCAMResult, Interval, Kline,
};
use std::collections::VecDeque;

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
enum Signal {
    Buy,
    Sell,
    Hold,
}

/// Trading decision with explanation
#[derive(Debug)]
struct TradingDecision {
    signal: Signal,
    confidence: f64,
    predicted_class: usize,
    explanation: String,
    important_periods: Vec<usize>,
    should_execute: bool,
    reason: String,
}

/// Simple trading strategy with interpretability
struct InterpretableStrategy {
    model: FinancialCNN,
    gradcam: GradCAM,
    processor: DataProcessor,
    confidence_threshold: f64,
    min_important_periods: usize,
}

impl InterpretableStrategy {
    fn new(sequence_length: usize, confidence_threshold: f64) -> Self {
        let config = CNNConfig {
            input_channels: 5,
            sequence_length,
            num_classes: 3,
            hidden_dims: vec![32, 64, 128],
        };

        Self {
            model: FinancialCNN::new(config),
            gradcam: GradCAM::new(1e-4, 0.5),
            processor: DataProcessor::new(sequence_length, 5),
            confidence_threshold,
            min_important_periods: 3,
        }
    }

    fn analyze(&mut self, klines: &[Kline]) -> Option<TradingDecision> {
        // Create features
        let features_batch = self.processor.create_single_sequence(klines)?;
        let features = features_batch.slice(ndarray::s![0, .., ..]).to_owned();

        // Run Grad-CAM
        let result = self.gradcam.compute(&mut self.model, &features, None);

        // Determine signal
        let signal = match result.predicted_class {
            0 => Signal::Sell,
            2 => Signal::Buy,
            _ => Signal::Hold,
        };

        // Generate explanation
        let explanation = self.gradcam.explain(&result);

        // Determine if we should execute
        let (should_execute, reason) = self.should_execute(&result, signal);

        Some(TradingDecision {
            signal,
            confidence: result.confidence,
            predicted_class: result.predicted_class,
            explanation,
            important_periods: result.important_periods.clone(),
            should_execute,
            reason,
        })
    }

    fn should_execute(&self, result: &GradCAMResult, signal: Signal) -> (bool, String) {
        // Check confidence threshold
        if result.confidence < self.confidence_threshold {
            return (
                false,
                format!(
                    "Confidence ({:.1}%) below threshold ({:.1}%)",
                    result.confidence * 100.0,
                    self.confidence_threshold * 100.0
                ),
            );
        }

        // Check if signal is not Hold
        if signal == Signal::Hold {
            return (false, "Model predicts neutral/hold".to_string());
        }

        // Check number of important periods
        if result.important_periods.len() < self.min_important_periods {
            return (
                false,
                format!(
                    "Too few important periods ({} < {})",
                    result.important_periods.len(),
                    self.min_important_periods
                ),
            );
        }

        // Check focus ratio (avoid extreme focus on single period)
        if result.focus_stats.peak_value > 0.95 {
            return (
                false,
                "Model overly focused on single period - potential overfitting".to_string(),
            );
        }

        (true, "All criteria met".to_string())
    }
}

fn main() {
    // Initialize logging
    env_logger::init();

    println!("Grad-CAM Finance - Trading Pipeline Example");
    println!("============================================\n");

    // Configuration
    let sequence_length = 60;
    let confidence_threshold = 0.4;
    let symbol = "BTCUSDT";

    // Create strategy
    let mut strategy = InterpretableStrategy::new(sequence_length, confidence_threshold);

    // Create Bybit client
    let client = BybitClient::new();

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Sequence length: {}", sequence_length);
    println!("  Confidence threshold: {:.1}%", confidence_threshold * 100.0);
    println!();

    // Fetch data
    println!("Fetching market data...");
    let klines = match client.get_klines(symbol, Interval::Hour1, Some(100), None, None) {
        Ok(k) => {
            println!("Fetched {} candles", k.len());
            k
        }
        Err(e) => {
            println!("Failed to fetch data: {}. Using synthetic data.", e);
            generate_synthetic_klines(100)
        }
    };

    // Get current price
    let current_price = klines.last().map(|k| k.close).unwrap_or(0.0);
    println!("Current price: ${:.2}\n", current_price);

    // Analyze with strategy
    println!("Running analysis...\n");

    match strategy.analyze(&klines) {
        Some(decision) => {
            print_decision(&decision);

            // Simulate what would happen
            println!("\n=== Simulation ===\n");
            if decision.should_execute {
                let action = match decision.signal {
                    Signal::Buy => "BUY",
                    Signal::Sell => "SELL",
                    Signal::Hold => "HOLD",
                };
                println!(
                    "Would execute: {} at ${:.2}",
                    action, current_price
                );

                // Calculate position size (example: 1% of portfolio)
                let portfolio_value = 100000.0;
                let position_size = portfolio_value * 0.01;
                let btc_amount = position_size / current_price;

                println!("Position size: ${:.2} ({:.6} BTC)", position_size, btc_amount);

                // Set stop loss and take profit
                let stop_loss = if decision.signal == Signal::Buy {
                    current_price * 0.98
                } else {
                    current_price * 1.02
                };
                let take_profit = if decision.signal == Signal::Buy {
                    current_price * 1.03
                } else {
                    current_price * 0.97
                };

                println!("Stop loss: ${:.2}", stop_loss);
                println!("Take profit: ${:.2}", take_profit);
            } else {
                println!("Trade NOT executed");
                println!("Reason: {}", decision.reason);
            }
        }
        None => {
            println!("Could not generate decision (insufficient data)");
        }
    }

    // Run multiple analyses to show distribution
    println!("\n=== Multiple Period Analysis ===\n");
    run_multiple_analyses(&mut strategy, &klines);
}

fn print_decision(decision: &TradingDecision) {
    println!("=== Trading Decision ===\n");

    let signal_str = match decision.signal {
        Signal::Buy => "BUY  [+]",
        Signal::Sell => "SELL [-]",
        Signal::Hold => "HOLD [=]",
    };

    println!("Signal: {}", signal_str);
    println!(
        "Prediction: {} (class {})",
        FinancialCNN::class_to_label(decision.predicted_class),
        decision.predicted_class
    );
    println!("Confidence: {:.1}%", decision.confidence * 100.0);
    println!();

    println!("Important periods: {:?}", decision.important_periods);
    println!();

    println!("Should execute: {}", if decision.should_execute { "YES" } else { "NO" });
    println!("Reason: {}", decision.reason);
    println!();

    println!("=== Model Explanation ===\n");
    println!("{}", decision.explanation);
}

fn run_multiple_analyses(strategy: &mut InterpretableStrategy, klines: &[Kline]) {
    let sequence_length = strategy.processor.sequence_length();
    let mut buy_count = 0;
    let mut sell_count = 0;
    let mut hold_count = 0;
    let mut executable_count = 0;

    let num_windows = klines.len().saturating_sub(sequence_length);

    println!("Analyzing {} time windows...\n", num_windows);

    for i in 0..num_windows {
        let window = &klines[i..i + sequence_length + 1];
        if let Some(decision) = strategy.analyze(window) {
            match decision.signal {
                Signal::Buy => buy_count += 1,
                Signal::Sell => sell_count += 1,
                Signal::Hold => hold_count += 1,
            }
            if decision.should_execute {
                executable_count += 1;
            }
        }
    }

    println!("Signal Distribution:");
    println!("  Buy:  {} ({:.1}%)", buy_count, buy_count as f64 / num_windows as f64 * 100.0);
    println!("  Sell: {} ({:.1}%)", sell_count, sell_count as f64 / num_windows as f64 * 100.0);
    println!("  Hold: {} ({:.1}%)", hold_count, hold_count as f64 / num_windows as f64 * 100.0);
    println!();
    println!(
        "Executable signals: {} ({:.1}%)",
        executable_count,
        executable_count as f64 / num_windows as f64 * 100.0
    );
}

fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut price = 50000.0;
    let mut klines = Vec::with_capacity(n);

    for i in 0..n {
        let trend = (i as f64 / 50.0).sin() * 0.005;
        let noise = rng.gen_range(-0.01..0.01);
        let change = trend + noise;

        let new_price = price * (1.0 + change);
        let volatility = rng.gen_range(0.005..0.015);

        let open = price;
        let close = new_price;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * volatility);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * volatility);
        let volume = rng.gen_range(100.0..1000.0);

        klines.push(Kline {
            timestamp: i as i64 * 3600000,
            open,
            high,
            low,
            close,
            volume,
            turnover: volume * close,
        });

        price = new_price;
    }

    klines
}
