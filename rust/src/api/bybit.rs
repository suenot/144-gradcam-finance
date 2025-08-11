//! Bybit API client for fetching cryptocurrency market data
//!
//! This module provides a client for interacting with Bybit's public API
//! to fetch OHLCV (Open, High, Low, Close, Volume) data.

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when interacting with the Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),
}

/// Kline/candlestick interval
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
    Month1,
}

impl Interval {
    /// Convert interval to API string
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Get interval duration in minutes
    pub fn minutes(&self) -> u64 {
        match self {
            Interval::Min1 => 1,
            Interval::Min3 => 3,
            Interval::Min5 => 5,
            Interval::Min15 => 15,
            Interval::Min30 => 30,
            Interval::Hour1 => 60,
            Interval::Hour2 => 120,
            Interval::Hour4 => 240,
            Interval::Hour6 => 360,
            Interval::Hour12 => 720,
            Interval::Day1 => 1440,
            Interval::Week1 => 10080,
            Interval::Month1 => 43200,
        }
    }
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Opening time (Unix timestamp in milliseconds)
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote currency volume)
    pub turnover: f64,
}

impl Kline {
    /// Get timestamp as DateTime<Utc>
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp).unwrap()
    }

    /// Calculate price return (close/open - 1)
    pub fn return_pct(&self) -> f64 {
        (self.close / self.open) - 1.0
    }

    /// Calculate price range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate typical price (high + low + close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if bullish candle (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate upper shadow
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower shadow
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }
}

/// API response structure for klines
#[derive(Debug, Deserialize)]
struct KlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: KlineResult,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create a new Bybit client with testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create client with custom base URL
    pub fn with_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval
    /// * `limit` - Number of candles to fetch (max 1000)
    /// * `start_time` - Optional start time (Unix timestamp in milliseconds)
    /// * `end_time` - Optional end time (Unix timestamp in milliseconds)
    ///
    /// # Returns
    /// Vector of Kline data sorted by timestamp ascending
    pub fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        limit: Option<u32>,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}",
            self.base_url,
            symbol,
            interval.as_str()
        );

        if let Some(l) = limit {
            url.push_str(&format!("&limit={}", l.min(1000)));
        }

        if let Some(st) = start_time {
            url.push_str(&format!("&start={}", st));
        }

        if let Some(et) = end_time {
            url.push_str(&format!("&end={}", et));
        }

        log::debug!("Fetching klines from: {}", url);

        let response: KlineResponse = self.client.get(&url).send()?.json()?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let mut klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp ascending (API returns descending)
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch historical klines with pagination
    ///
    /// This method fetches data in chunks to get more than 1000 candles
    pub fn get_klines_history(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut all_klines = Vec::new();
        let mut current_end = end_time;
        let interval_ms = interval.minutes() as i64 * 60 * 1000;

        while current_end > start_time {
            let klines = self.get_klines(symbol, interval, Some(1000), None, Some(current_end))?;

            if klines.is_empty() {
                break;
            }

            let oldest_timestamp = klines.first().map(|k| k.timestamp).unwrap_or(start_time);

            // Filter klines within our range
            let filtered: Vec<Kline> = klines
                .into_iter()
                .filter(|k| k.timestamp >= start_time && k.timestamp <= end_time)
                .collect();

            all_klines.extend(filtered);

            // Move window back
            current_end = oldest_timestamp - interval_ms;

            // Small delay to avoid rate limiting
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Remove duplicates and sort
        all_klines.sort_by_key(|k| k.timestamp);
        all_klines.dedup_by_key(|k| k.timestamp);

        Ok(all_klines)
    }

    /// Get ticker information for a symbol
    pub fn get_ticker(&self, symbol: &str) -> Result<TickerInfo, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: TickerResponse = self.client.get(&url).send()?.json()?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::ApiError {
                code: -1,
                message: "No ticker data found".to_string(),
            })
    }
}

/// Ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

impl TickerInfo {
    /// Get last price as f64
    pub fn last_price_f64(&self) -> Option<f64> {
        self.last_price.parse().ok()
    }

    /// Get 24h price change percentage as f64
    pub fn price_change_pct(&self) -> Option<f64> {
        self.price_24h_pcnt.parse().ok()
    }
}

#[derive(Debug, Deserialize)]
struct TickerResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: TickerResult,
}

#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_as_str() {
        assert_eq!(Interval::Min1.as_str(), "1");
        assert_eq!(Interval::Hour1.as_str(), "60");
        assert_eq!(Interval::Day1.as_str(), "D");
    }

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!((kline.return_pct() - 0.05).abs() < 1e-10);
        assert!((kline.range() - 15.0).abs() < 1e-10);
        assert!((kline.typical_price() - 103.333333).abs() < 0.001);
        assert!(kline.is_bullish());
        assert!((kline.body_size() - 5.0).abs() < 1e-10);
        assert!((kline.upper_shadow() - 5.0).abs() < 1e-10);
        assert!((kline.lower_shadow() - 5.0).abs() < 1e-10);
    }
}
