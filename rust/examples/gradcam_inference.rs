//! Example: Grad-CAM Inference
//!
//! This example demonstrates how to use Grad-CAM to interpret CNN predictions
//! on financial time series data.
//!
//! Run with: cargo run --example gradcam_inference

use gradcam_finance::{
    BybitClient, CNNConfig, DataProcessor, FinancialCNN, GradCAM, Interval,
};
use ndarray::Array2;

fn main() {
    // Initialize logging
    env_logger::init();

    println!("Grad-CAM Finance - Inference Example");
    println!("=====================================\n");

    // Configuration
    let sequence_length = 60;
    let config = CNNConfig {
        input_channels: 5,
        sequence_length,
        num_classes: 3,
        hidden_dims: vec![32, 64, 128],
    };

    // Create model and Grad-CAM
    let mut model = FinancialCNN::new(config);
    let gradcam = GradCAM::new(1e-4, 0.5);
    let processor = DataProcessor::new(sequence_length, 5);

    println!("Model configuration:");
    println!("  Input channels: {}", model.config().input_channels);
    println!("  Sequence length: {}", model.config().sequence_length);
    println!("  Number of classes: {}", model.config().num_classes);
    println!();

    // Try to fetch real data
    println!("Fetching data from Bybit...");
    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None) {
        Ok(k) => {
            println!("Fetched {} candles\n", k.len());
            k
        }
        Err(e) => {
            println!("Failed to fetch data: {}. Using synthetic data.\n", e);
            generate_synthetic_klines(100)
        }
    };

    // Create feature sequence
    println!("Creating feature sequence...");
    let features_batch = processor.create_single_sequence(&klines);

    if features_batch.is_none() {
        println!("Not enough data for sequence. Need at least {} candles.", sequence_length);
        return;
    }

    let features_batch = features_batch.unwrap();
    let features = features_batch.slice(ndarray::s![0, .., ..]).to_owned();

    println!("Feature shape: {:?}\n", features.dim());

    // Run inference with Grad-CAM
    println!("Running Grad-CAM inference...");
    let result = gradcam.compute(&mut model, &features, None);

    // Display results
    println!("\n=== Grad-CAM Results ===\n");
    println!(
        "Prediction: {} (class {})",
        FinancialCNN::class_to_label(result.predicted_class),
        result.predicted_class
    );
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    println!();

    println!("Focus Statistics:");
    println!("  Peak attention period: {} (value: {:.3})",
             result.focus_stats.peak_period,
             result.focus_stats.peak_value);
    println!("  Recent focus (last 10): {:.3}", result.focus_stats.recent_focus);
    println!("  Historical focus (first 10): {:.3}", result.focus_stats.historical_focus);
    println!("  Focus ratio: {:.2}x", result.focus_stats.focus_ratio);
    println!();

    println!("Important periods (attention > 0.5): {:?}", result.important_periods);
    println!();

    // Print heatmap visualization
    println!("Attention Heatmap (ASCII visualization):");
    print_heatmap(&result.heatmap);
    println!();

    // Print explanation
    println!("=== Explanation ===\n");
    println!("{}", gradcam.explain(&result));

    // Analyze what the model focused on
    println!("=== Analysis ===\n");
    analyze_focus(&klines, &result, sequence_length);
}

/// Generate synthetic klines for testing
fn generate_synthetic_klines(n: usize) -> Vec<gradcam_finance::Kline> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut price = 50000.0; // Starting price for BTC-like data
    let mut klines = Vec::with_capacity(n);

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let new_price = price * (1.0 + change);
        let volatility = rng.gen_range(0.005..0.015);

        let open = price;
        let close = new_price;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * volatility);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * volatility);
        let volume = rng.gen_range(100.0..1000.0);

        klines.push(gradcam_finance::Kline {
            timestamp: i as i64 * 3600000, // 1 hour intervals
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

/// Print ASCII heatmap
fn print_heatmap(heatmap: &[f64]) {
    let chars = [' ', '.', ':', 'o', 'O', '@'];
    let width = 60;

    // Downsample if needed
    let step = (heatmap.len() as f64 / width as f64).ceil() as usize;

    print!("    ");
    for i in (0..width).step_by(10) {
        print!("{:<10}", i * step);
    }
    println!();

    print!("    ");
    for i in 0..width {
        let idx = (i * step).min(heatmap.len() - 1);
        let val = heatmap[idx];
        let char_idx = ((val * (chars.len() - 1) as f64).round() as usize).min(chars.len() - 1);
        print!("{}", chars[char_idx]);
    }
    println!();

    println!("\n    Legend: ' '=0.0  '.'=0.2  ':'=0.4  'o'=0.6  'O'=0.8  '@'=1.0");
}

/// Analyze what the model focused on
fn analyze_focus(
    klines: &[gradcam_finance::Kline],
    result: &gradcam_finance::GradCAMResult,
    sequence_length: usize,
) {
    let start_idx = klines.len().saturating_sub(sequence_length);
    let window = &klines[start_idx..];

    // Find the most important period
    let peak = result.focus_stats.peak_period;
    if peak < window.len() {
        let kline = &window[peak];
        println!("At peak attention period ({}):", peak);
        println!("  Price: ${:.2} -> ${:.2}", kline.open, kline.close);
        println!("  Change: {:.2}%", kline.return_pct() * 100.0);
        println!("  Range: ${:.2}", kline.range());
        println!("  Candle type: {}", if kline.is_bullish() { "Bullish" } else { "Bearish" });
    }

    // Analyze recent price action
    let recent = &window[window.len().saturating_sub(5)..];
    let recent_returns: Vec<f64> = recent.iter().map(|k| k.return_pct()).collect();
    let avg_recent_return = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;

    println!();
    println!("Recent price action (last 5 periods):");
    println!("  Average return: {:.2}%", avg_recent_return * 100.0);
    println!(
        "  Bullish/Bearish: {}/{}",
        recent.iter().filter(|k| k.is_bullish()).count(),
        recent.iter().filter(|k| !k.is_bullish()).count()
    );

    // Model focus interpretation
    println!();
    if result.focus_stats.focus_ratio > 1.5 {
        println!("Interpretation: Model is focusing on recent momentum.");
        if avg_recent_return > 0.0 && result.predicted_class == 2 {
            println!("  -> Positive momentum detected, predicting continuation.");
        } else if avg_recent_return < 0.0 && result.predicted_class == 0 {
            println!("  -> Negative momentum detected, predicting continuation.");
        }
    } else if result.focus_stats.focus_ratio < 0.67 {
        println!("Interpretation: Model is focusing on historical patterns.");
        println!("  -> May be detecting support/resistance or reversal patterns.");
    } else {
        println!("Interpretation: Model attention is balanced.");
        println!("  -> Considering both recent and historical context.");
    }
}
