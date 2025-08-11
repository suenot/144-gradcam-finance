//! Example: Fetching data from Bybit
//!
//! This example demonstrates how to fetch OHLCV data from the Bybit exchange.
//!
//! Run with: cargo run --example fetch_data

use gradcam_finance::{BybitClient, Interval};

fn main() {
    // Initialize logging
    env_logger::init();

    println!("Grad-CAM Finance - Data Fetching Example");
    println!("=========================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch ticker information
    println!("Fetching BTCUSDT ticker...");
    match client.get_ticker("BTCUSDT") {
        Ok(ticker) => {
            println!("Symbol: {}", ticker.symbol);
            println!("Last Price: ${}", ticker.last_price);
            println!("24h High: ${}", ticker.high_price_24h);
            println!("24h Low: ${}", ticker.low_price_24h);
            println!("24h Volume: {}", ticker.volume_24h);
            println!("24h Change: {}%", ticker.price_24h_pcnt);
        }
        Err(e) => {
            eprintln!("Failed to fetch ticker: {}", e);
        }
    }

    println!("\n-----------------------------------\n");

    // Fetch recent klines
    println!("Fetching recent 1-hour candles...");
    match client.get_klines("BTCUSDT", Interval::Hour1, Some(10), None, None) {
        Ok(klines) => {
            println!("Fetched {} candles:\n", klines.len());

            println!(
                "{:>20} {:>12} {:>12} {:>12} {:>12} {:>15}",
                "Time (UTC)", "Open", "High", "Low", "Close", "Volume"
            );
            println!("{}", "-".repeat(90));

            for kline in &klines {
                let datetime = kline.datetime();
                println!(
                    "{:>20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
                    datetime.format("%Y-%m-%d %H:%M"),
                    kline.open,
                    kline.high,
                    kline.low,
                    kline.close,
                    kline.volume
                );
            }

            // Calculate some statistics
            println!("\n-----------------------------------\n");
            println!("Statistics:");

            let returns: Vec<f64> = klines.windows(2).map(|w| w[1].return_pct()).collect();
            let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let avg_range = klines.iter().map(|k| k.range()).sum::<f64>() / klines.len() as f64;

            println!("Average return: {:.4}%", avg_return * 100.0);
            println!("Average range: ${:.2}", avg_range);

            let bullish = klines.iter().filter(|k| k.is_bullish()).count();
            println!("Bullish candles: {}/{}", bullish, klines.len());
        }
        Err(e) => {
            eprintln!("Failed to fetch klines: {}", e);
        }
    }

    println!("\n-----------------------------------\n");

    // Fetch multiple symbols
    let symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT"];
    println!("Fetching multiple symbols...\n");

    for symbol in &symbols {
        match client.get_ticker(symbol) {
            Ok(ticker) => {
                if let Some(price) = ticker.last_price_f64() {
                    if let Some(change) = ticker.price_change_pct() {
                        let direction = if change >= 0.0 { "+" } else { "" };
                        println!("{}: ${:.2} ({}{}%)", symbol, price, direction, change * 100.0);
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: Error - {}", symbol, e);
            }
        }
    }

    println!("\nDone!");
}
