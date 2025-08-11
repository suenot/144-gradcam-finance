//! # Grad-CAM for Financial Markets
//!
//! This library provides implementations of Grad-CAM (Gradient-weighted Class Activation Mapping)
//! for interpreting CNN-based trading decisions, with Bybit exchange integration.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data processing and feature engineering
//! - `models` - CNN inference and Grad-CAM computation
//!
//! ## Example
//!
//! ```rust,no_run
//! use gradcam_finance::{BybitClient, Interval, DataProcessor};
//!
//! // Fetch data from Bybit
//! let client = BybitClient::new();
//! let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;
//!
//! // Process data
//! let processor = DataProcessor::new();
//! let features = processor.create_features(&klines);
//! ```

pub mod api;
pub mod data;
pub mod models;

pub use api::bybit::{BybitClient, BybitError, Interval, Kline, TickerInfo};
pub use data::features::FeatureEngineering;
pub use data::processor::DataProcessor;
pub use models::cnn::{CNNConfig, FinancialCNN};
pub use models::gradcam::{GradCAM, GradCAMResult};
