# Grad-CAM Finance - Rust Implementation

High-performance Rust implementation of Grad-CAM for financial CNN interpretability with Bybit exchange integration.

## Features

- **Bybit API Client**: Fetch real-time and historical OHLCV data
- **Data Processing**: Feature engineering and sequence preparation for CNN input
- **CNN Inference**: Financial time series classification
- **Grad-CAM**: Model interpretability with attention heatmaps
- **Trading Pipeline**: Complete example of interpretable trading decisions

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gradcam_finance = { path = "." }
```

## Quick Start

### Fetching Data from Bybit

```rust
use gradcam_finance::{BybitClient, Interval};

let client = BybitClient::new();

// Get ticker info
let ticker = client.get_ticker("BTCUSDT")?;
println!("BTC Price: ${}", ticker.last_price);

// Get historical candles
let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;
```

### Running Grad-CAM Inference

```rust
use gradcam_finance::{CNNConfig, DataProcessor, FinancialCNN, GradCAM};

// Create model
let config = CNNConfig::default();
let mut model = FinancialCNN::new(config);

// Create Grad-CAM
let gradcam = GradCAM::default();

// Process data
let processor = DataProcessor::default();
let features = processor.create_single_sequence(&klines).unwrap();

// Run inference with explanation
let result = gradcam.compute(&mut model, &features.slice(s![0, .., ..]).to_owned(), None);

println!("Prediction: {}", FinancialCNN::class_to_label(result.predicted_class));
println!("Confidence: {:.1}%", result.confidence * 100.0);
println!("Important periods: {:?}", result.important_periods);
```

## Examples

Run the examples with:

```bash
# Fetch data from Bybit
cargo run --example fetch_data

# Run Grad-CAM inference
cargo run --example gradcam_inference

# Full trading pipeline
cargo run --example trading_pipeline
```

## Module Structure

```
src/
├── lib.rs              # Main library entry point
├── api/
│   ├── mod.rs
│   └── bybit.rs        # Bybit API client
├── data/
│   ├── mod.rs
│   ├── features.rs     # Technical indicators and feature engineering
│   └── processor.rs    # Data processing and sequence creation
└── models/
    ├── mod.rs
    ├── cnn.rs          # CNN model for inference
    └── gradcam.rs      # Grad-CAM implementation
```

## API Reference

### BybitClient

```rust
// Create client
let client = BybitClient::new();           // Production
let client = BybitClient::testnet();       // Testnet

// Fetch klines
let klines = client.get_klines(
    "BTCUSDT",           // Symbol
    Interval::Hour1,     // Interval
    Some(100),           // Limit (max 1000)
    None,                // Start time (ms)
    None,                // End time (ms)
)?;

// Fetch historical data with pagination
let history = client.get_klines_history(
    "BTCUSDT",
    Interval::Hour1,
    start_timestamp,
    end_timestamp,
)?;
```

### DataProcessor

```rust
let processor = DataProcessor::new(60, 5);  // 60 periods, 5 channels (OHLCV)

// Create single sequence for inference
let features = processor.create_single_sequence(&klines);

// Create multiple sequences (sliding window)
let sequences = processor.create_sequences(&klines);

// Create labels for training
let labels = processor.create_labels(&klines, 0.005, 1);  // 0.5% threshold, 1 period lookahead
```

### FeatureEngineering

```rust
use gradcam_finance::FeatureEngineering;

// Technical indicators
let sma = FeatureEngineering::sma(&prices, 20);
let ema = FeatureEngineering::ema(&prices, 12);
let rsi = FeatureEngineering::rsi(&prices, 14);
let (macd, signal, histogram) = FeatureEngineering::macd(&prices, 12, 26, 9);
let (upper, middle, lower) = FeatureEngineering::bollinger_bands(&prices, 20, 2.0);
let atr = FeatureEngineering::atr(&klines, 14);

// Normalization
let normalized = FeatureEngineering::normalize(&values);
let standardized = FeatureEngineering::standardize(&values);

// Returns
let returns = FeatureEngineering::returns(&prices);
let log_returns = FeatureEngineering::log_returns(&prices);
```

### GradCAM

```rust
let gradcam = GradCAM::new(
    1e-4,   // epsilon for numerical differentiation
    0.5,    // threshold for important periods
);

let result = gradcam.compute(&mut model, &features, None);

// Result contains:
// - heatmap: Vec<f64> - attention values for each time step
// - predicted_class: usize - predicted class (0=down, 1=neutral, 2=up)
// - confidence: f64 - prediction confidence
// - important_periods: Vec<usize> - periods with attention > threshold
// - focus_stats: FocusStats - analysis of attention distribution

// Generate explanation
let explanation = gradcam.explain(&result);
```

## Note on Training

This Rust implementation focuses on **inference** rather than training. For training CNN models, we recommend using the Python implementation with PyTorch:

1. Train your model in Python using `python/train.py`
2. Export the trained weights
3. Load weights in Rust for production inference

For production use with trained PyTorch models, consider using:
- [tch-rs](https://github.com/LaurentMazare/tch-rs) - PyTorch bindings for Rust
- [candle](https://github.com/huggingface/candle) - Minimalist ML framework by Hugging Face

## License

MIT License
