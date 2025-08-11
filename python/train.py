"""
Training Pipeline for Financial CNN with Grad-CAM

This module provides a complete training pipeline for the Financial CNN model,
including data loading, preprocessing, training, and evaluation.

Functions:
    - fetch_bybit_data: Fetch OHLCV data from Bybit exchange
    - prepare_dataset: Prepare dataset for training
    - train_model: Train the CNN model
    - evaluate_model: Evaluate model performance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
import json
import os

from model import FinancialCNN, GradCAM, create_feature_tensor


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial time series.

    Args:
        features: Feature array of shape (num_samples, channels, sequence_length)
        labels: Label array of shape (num_samples,)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def fetch_bybit_data(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Fetch OHLCV data from Bybit exchange.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Candlestick interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limit: Number of candles to fetch (max 1000)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds

    Returns:
        OHLCV data as numpy array or None if request fails
    """
    try:
        import requests
    except ImportError:
        print("requests library required. Install with: pip install requests")
        return None

    base_url = "https://api.bybit.com"
    endpoint = "/v5/market/kline"

    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)
    }

    if start_time:
        params["start"] = start_time
    if end_time:
        params["end"] = end_time

    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            print(f"API Error: {data.get('retMsg')}")
            return None

        klines = data["result"]["list"]

        # Convert to numpy array
        # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
        ohlcv = []
        for kline in klines:
            ohlcv.append([
                float(kline[1]),  # Open
                float(kline[2]),  # High
                float(kline[3]),  # Low
                float(kline[4]),  # Close
                float(kline[5])   # Volume
            ])

        # Bybit returns data in descending order, reverse it
        ohlcv = np.array(ohlcv[::-1])

        return ohlcv

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def fetch_yahoo_data(
    symbol: str = "AAPL",
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)

    Returns:
        OHLCV data as numpy array or None if request fails
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance library required. Install with: pip install yfinance")
        return None

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print(f"No data found for {symbol}")
            return None

        # Extract OHLCV columns
        ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

        return ohlcv

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None


def create_labels(
    prices: np.ndarray,
    threshold: float = 0.0,
    lookahead: int = 1
) -> np.ndarray:
    """
    Create classification labels based on future price movement.

    Args:
        prices: Close prices array
        threshold: Threshold for neutral class (e.g., 0.01 for 1%)
        lookahead: Number of periods to look ahead

    Returns:
        Labels array: 0 = down, 1 = neutral, 2 = up
    """
    returns = np.zeros(len(prices))
    returns[:-lookahead] = (prices[lookahead:] - prices[:-lookahead]) / prices[:-lookahead]

    labels = np.ones(len(prices), dtype=np.int64)  # Default to neutral
    labels[returns < -threshold] = 0  # Down
    labels[returns > threshold] = 2   # Up

    return labels


def prepare_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = 60,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for training.

    Args:
        data: OHLCV data array of shape (num_periods, 5)
        labels: Labels array
        sequence_length: Length of each sequence
        normalize: Whether to normalize features

    Returns:
        Tuple of (features, labels) arrays
    """
    num_samples = len(data) - sequence_length

    features = []
    valid_labels = []

    for i in range(num_samples):
        seq = data[i:i + sequence_length].copy()

        if normalize:
            # Normalize prices relative to first close
            price_scale = seq[0, 3]  # First close price
            seq[:, :4] = seq[:, :4] / price_scale - 1

            # Normalize volume
            vol_mean = seq[:, 4].mean()
            vol_std = seq[:, 4].std() + 1e-8
            seq[:, 4] = (seq[:, 4] - vol_mean) / vol_std

        # Transpose to (channels, length)
        features.append(seq.T)
        valid_labels.append(labels[i + sequence_length])

    return np.array(features), np.array(valid_labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
    patience: int = 10,
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Train the CNN model.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        patience: Early stopping patience
        save_path: Path to save best model

    Returns:
        Dictionary with training history
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_path)
                print(f"  Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()

    # Per-class metrics
    metrics = {'accuracy': accuracy}

    for cls in range(3):
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == cls).mean()
            metrics[f'class_{cls}_accuracy'] = cls_acc

    # Precision, Recall, F1 for each class
    for cls in range(3):
        tp = ((all_preds == cls) & (all_labels == cls)).sum()
        fp = ((all_preds == cls) & (all_labels != cls)).sum()
        fn = ((all_preds != cls) & (all_labels == cls)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f'class_{cls}_precision'] = precision
        metrics[f'class_{cls}_recall'] = recall
        metrics[f'class_{cls}_f1'] = f1

    return metrics


def generate_synthetic_data(
    num_samples: int = 1000,
    sequence_length: int = 60,
    num_channels: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic financial data for testing.

    Args:
        num_samples: Number of samples
        sequence_length: Length of each sequence
        num_channels: Number of features (OHLCV = 5)

    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(42)

    features = []
    labels = []

    for _ in range(num_samples):
        # Generate base price movement
        label = np.random.randint(0, 3)  # 0: down, 1: neutral, 2: up

        # Create price series with trend
        base_price = 100
        trend = (label - 1) * 0.001  # Negative, zero, or positive trend
        noise = np.random.randn(sequence_length) * 0.01

        close_prices = base_price * np.cumprod(1 + trend + noise)

        # Generate OHLCV from close prices
        ohlcv = np.zeros((sequence_length, num_channels))
        ohlcv[:, 3] = close_prices  # Close

        # Generate open, high, low from close
        for i in range(sequence_length):
            volatility = np.random.uniform(0.005, 0.02)
            ohlcv[i, 0] = close_prices[i] * (1 + np.random.randn() * volatility * 0.5)  # Open
            ohlcv[i, 1] = max(ohlcv[i, 0], close_prices[i]) * (1 + abs(np.random.randn()) * volatility)  # High
            ohlcv[i, 2] = min(ohlcv[i, 0], close_prices[i]) * (1 - abs(np.random.randn()) * volatility)  # Low
            ohlcv[i, 4] = np.random.uniform(1000, 10000)  # Volume

        # Normalize
        price_scale = ohlcv[0, 3]
        ohlcv[:, :4] = ohlcv[:, :4] / price_scale - 1
        ohlcv[:, 4] = (ohlcv[:, 4] - ohlcv[:, 4].mean()) / (ohlcv[:, 4].std() + 1e-8)

        features.append(ohlcv.T)  # Shape: (channels, sequence_length)
        labels.append(label)

    return np.array(features), np.array(labels)


def run_training_pipeline(
    data_source: str = "synthetic",
    symbol: str = "BTCUSDT",
    sequence_length: int = 60,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    save_dir: str = "checkpoints"
):
    """
    Run the complete training pipeline.

    Args:
        data_source: 'synthetic', 'bybit', or 'yahoo'
        symbol: Trading symbol
        sequence_length: Sequence length
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
    """
    print(f"Running training pipeline with {data_source} data...")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load or generate data
    if data_source == "synthetic":
        print("Generating synthetic data...")
        features, labels = generate_synthetic_data(num_samples=2000, sequence_length=sequence_length)
    elif data_source == "bybit":
        print(f"Fetching data from Bybit for {symbol}...")
        ohlcv = fetch_bybit_data(symbol=symbol, interval="60", limit=1000)
        if ohlcv is None:
            print("Failed to fetch data. Using synthetic data instead.")
            features, labels = generate_synthetic_data(num_samples=2000, sequence_length=sequence_length)
        else:
            raw_labels = create_labels(ohlcv[:, 3], threshold=0.005, lookahead=1)
            features, labels = prepare_sequences(ohlcv, raw_labels, sequence_length)
    elif data_source == "yahoo":
        print(f"Fetching data from Yahoo Finance for {symbol}...")
        ohlcv = fetch_yahoo_data(symbol=symbol)
        if ohlcv is None:
            print("Failed to fetch data. Using synthetic data instead.")
            features, labels = generate_synthetic_data(num_samples=2000, sequence_length=sequence_length)
        else:
            raw_labels = create_labels(ohlcv[:, 3], threshold=0.005, lookahead=1)
            features, labels = prepare_sequences(ohlcv, raw_labels, sequence_length)
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    print(f"Data shape: {features.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Split data
    num_samples = len(features)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    train_features = features[:train_size]
    train_labels = labels[:train_size]
    val_features = features[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    test_features = features[train_size + val_size:]
    test_labels = labels[train_size + val_size:]

    # Create data loaders
    train_dataset = FinancialDataset(train_features, train_labels)
    val_dataset = FinancialDataset(val_features, val_labels)
    test_dataset = FinancialDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    model = FinancialCNN(
        input_channels=5,
        num_classes=3,
        sequence_length=sequence_length
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    save_path = os.path.join(save_dir, "best_model.pt")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        patience=10,
        save_path=save_path
    )

    # Load best model and evaluate
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, device)
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return model, history, metrics


if __name__ == "__main__":
    # Run training pipeline with synthetic data
    model, history, metrics = run_training_pipeline(
        data_source="synthetic",
        sequence_length=60,
        batch_size=32,
        num_epochs=20,
        learning_rate=0.001
    )
