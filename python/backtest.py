"""
Backtesting Framework with Interpretable Signals

This module provides a backtesting framework that integrates Grad-CAM
for explainable trading decisions.

Classes:
    - Trade: Represents a single trade with explanation
    - BacktestEngine: Main backtesting engine
    - PerformanceMetrics: Calculate trading performance metrics
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from model import FinancialCNN, GradCAM, create_feature_tensor


class Signal(Enum):
    """Trading signal types."""
    BUY = 1
    HOLD = 0
    SELL = -1


@dataclass
class Trade:
    """
    Represents a single trade with Grad-CAM explanation.

    Attributes:
        entry_time: Trade entry timestamp
        entry_price: Entry price
        exit_time: Trade exit timestamp (if closed)
        exit_price: Exit price (if closed)
        direction: 1 for long, -1 for short
        size: Position size
        pnl: Profit/Loss
        confidence: Model confidence for the trade
        gradcam_explanation: Grad-CAM heatmap for the entry decision
        important_periods: Time periods that influenced the decision
    """
    entry_time: int
    entry_price: float
    direction: int
    size: float = 1.0
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    confidence: float = 0.0
    gradcam_explanation: Optional[np.ndarray] = None
    important_periods: List[int] = field(default_factory=list)

    def close(self, exit_time: int, exit_price: float):
        """Close the trade and calculate PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.pnl = self.direction * (exit_price - self.entry_price) / self.entry_price * self.size

    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None

    @property
    def is_profitable(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl > 0


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Attributes:
        trades: List of all trades
        equity_curve: Equity curve over time
        returns: Period returns
        metrics: Performance metrics dictionary
    """
    trades: List[Trade]
    equity_curve: np.ndarray
    returns: np.ndarray
    metrics: Dict[str, float]


class PerformanceMetrics:
    """Calculate trading performance metrics."""

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array of period returns
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of trading periods per year

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sortino ratio.

        Args:
            returns: Array of period returns
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of trading periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve array

        Returns:
            Maximum drawdown as a positive percentage
        """
        if len(equity_curve) == 0:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return drawdown.max()

    @staticmethod
    def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Array of period returns
            equity_curve: Equity curve array
            periods_per_year: Number of trading periods per year

        Returns:
            Calmar ratio
        """
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)
        if max_dd == 0:
            return float('inf') if returns.mean() > 0 else 0.0

        annual_return = (1 + returns.mean()) ** periods_per_year - 1
        return annual_return / max_dd

    @staticmethod
    def win_rate(trades: List[Trade]) -> float:
        """Calculate win rate of trades."""
        if len(trades) == 0:
            return 0.0

        winning = sum(1 for t in trades if t.pnl > 0)
        return winning / len(trades)

    @staticmethod
    def profit_factor(trades: List[Trade]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def average_trade(trades: List[Trade]) -> float:
        """Calculate average trade PnL."""
        if len(trades) == 0:
            return 0.0
        return np.mean([t.pnl for t in trades])

    @staticmethod
    def calculate_all(
        trades: List[Trade],
        returns: np.ndarray,
        equity_curve: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        return {
            'total_trades': len(trades),
            'win_rate': PerformanceMetrics.win_rate(trades),
            'profit_factor': PerformanceMetrics.profit_factor(trades),
            'average_trade': PerformanceMetrics.average_trade(trades),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(equity_curve),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, equity_curve),
            'total_return': (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 0 else 0.0
        }


class BacktestEngine:
    """
    Backtesting engine with Grad-CAM explanations.

    This engine runs a trading strategy using a trained CNN model and
    provides Grad-CAM explanations for each trading decision.

    Args:
        model: Trained FinancialCNN model
        sequence_length: Input sequence length
        confidence_threshold: Minimum confidence to take a trade
        cam_threshold: Minimum CAM value to consider a period important
    """

    def __init__(
        self,
        model: FinancialCNN,
        sequence_length: int = 60,
        confidence_threshold: float = 0.6,
        cam_threshold: float = 0.5
    ):
        self.model = model
        self.model.eval()
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.cam_threshold = cam_threshold
        self.gradcam = GradCAM(model)

    def generate_signal(
        self,
        features: np.ndarray
    ) -> Tuple[Signal, float, Optional[np.ndarray], List[int]]:
        """
        Generate trading signal with explanation.

        Args:
            features: OHLCV data of shape (sequence_length, 5)

        Returns:
            Tuple of (signal, confidence, gradcam_heatmap, important_periods)
        """
        # Create input tensor
        input_tensor = create_feature_tensor(features)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        # Map prediction to signal
        signal_map = {0: Signal.SELL, 1: Signal.HOLD, 2: Signal.BUY}
        signal = signal_map[pred_class]

        # Get Grad-CAM explanation
        gradcam_heatmap = None
        important_periods = []

        if confidence >= self.confidence_threshold and signal != Signal.HOLD:
            # Need to recreate tensor with gradients
            input_tensor = create_feature_tensor(features)
            input_tensor.requires_grad_(True)
            gradcam_heatmap = self.gradcam(input_tensor, target_class=pred_class)

            # Find important periods
            important_periods = np.where(gradcam_heatmap > self.cam_threshold)[0].tolist()

        return signal, confidence, gradcam_heatmap, important_periods

    def run(
        self,
        data: np.ndarray,
        initial_capital: float = 100000.0,
        position_size: float = 0.1,
        transaction_cost: float = 0.001
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV data array of shape (num_periods, 5)
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            transaction_cost: Transaction cost as fraction

        Returns:
            BacktestResult with trades and performance metrics
        """
        trades: List[Trade] = []
        equity_curve = [initial_capital]
        returns = []

        capital = initial_capital
        current_trade: Optional[Trade] = None

        # Iterate through data
        for i in range(self.sequence_length, len(data) - 1):
            # Get current window
            window = data[i - self.sequence_length:i]
            current_price = data[i, 3]  # Close price
            next_price = data[i + 1, 3]  # Next close for exit

            # Generate signal
            signal, confidence, cam, important_periods = self.generate_signal(window)

            # Handle existing position
            if current_trade is not None:
                # Check exit conditions
                should_exit = False

                # Exit on opposite signal
                if signal == Signal.BUY and current_trade.direction == -1:
                    should_exit = True
                elif signal == Signal.SELL and current_trade.direction == 1:
                    should_exit = True

                if should_exit:
                    current_trade.close(i, current_price)
                    capital += current_trade.pnl * capital - abs(capital * position_size * transaction_cost)
                    trades.append(current_trade)
                    current_trade = None

            # Open new position
            if current_trade is None and signal != Signal.HOLD:
                if confidence >= self.confidence_threshold:
                    direction = 1 if signal == Signal.BUY else -1
                    current_trade = Trade(
                        entry_time=i,
                        entry_price=current_price,
                        direction=direction,
                        size=position_size,
                        confidence=confidence,
                        gradcam_explanation=cam,
                        important_periods=important_periods
                    )
                    capital -= abs(capital * position_size * transaction_cost)

            # Calculate period return
            if current_trade is not None:
                period_return = current_trade.direction * (next_price - current_price) / current_price * position_size
            else:
                period_return = 0.0

            returns.append(period_return)
            capital = capital * (1 + period_return)
            equity_curve.append(capital)

        # Close any remaining position
        if current_trade is not None:
            current_trade.close(len(data) - 1, data[-1, 3])
            trades.append(current_trade)

        # Calculate metrics
        returns = np.array(returns)
        equity_curve = np.array(equity_curve)
        metrics = PerformanceMetrics.calculate_all(trades, returns, equity_curve)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics
        )

    def explain_trade(self, trade: Trade, data: np.ndarray) -> Dict:
        """
        Generate detailed explanation for a trade.

        Args:
            trade: Trade to explain
            data: Full OHLCV data array

        Returns:
            Dictionary with trade explanation
        """
        window = data[trade.entry_time - self.sequence_length:trade.entry_time]

        explanation = {
            'entry_time': trade.entry_time,
            'direction': 'LONG' if trade.direction == 1 else 'SHORT',
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'confidence': trade.confidence,
            'important_periods': trade.important_periods,
            'num_important_periods': len(trade.important_periods),
        }

        if trade.gradcam_explanation is not None:
            # Analyze what the model focused on
            cam = trade.gradcam_explanation

            # Find peak attention period
            peak_period = cam.argmax()
            explanation['peak_attention_period'] = int(peak_period)
            explanation['peak_attention_value'] = float(cam[peak_period])

            # Analyze if model focused on recent or historical data
            recent_attention = cam[-10:].mean()
            historical_attention = cam[:-10].mean()
            explanation['recent_focus'] = float(recent_attention)
            explanation['historical_focus'] = float(historical_attention)
            explanation['focus_ratio'] = float(recent_attention / (historical_attention + 1e-8))

        return explanation


def run_backtest_example():
    """Run example backtest with synthetic data."""
    from train import generate_synthetic_data, FinancialDataset

    print("Running backtest example...")

    # Generate synthetic data
    print("Generating synthetic data...")
    features, labels = generate_synthetic_data(num_samples=500, sequence_length=60)

    # Reconstruct OHLCV data from features
    # Note: This is a simplified reconstruction for demonstration
    ohlcv_data = []
    base_price = 100

    for i in range(500):
        # Generate price series
        trend = (labels[i] - 1) * 0.001
        for j in range(60):
            noise = np.random.randn() * 0.01
            price = base_price * (1 + trend + noise)
            volatility = 0.01
            ohlcv_data.append([
                price * (1 + np.random.randn() * volatility * 0.5),  # Open
                price * (1 + abs(np.random.randn()) * volatility),   # High
                price * (1 - abs(np.random.randn()) * volatility),   # Low
                price,                                                 # Close
                np.random.uniform(1000, 10000)                        # Volume
            ])
            base_price = price

    ohlcv_data = np.array(ohlcv_data)
    print(f"Data shape: {ohlcv_data.shape}")

    # Create and train a simple model
    print("Creating model...")
    model = FinancialCNN(input_channels=5, num_classes=3, sequence_length=60)

    # Quick training on subset
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for i in range(0, min(100, len(features)), 10):
            batch_features = torch.FloatTensor(features[i:i+10])
            batch_labels = torch.LongTensor(labels[i:i+10])

            optimizer.zero_grad()
            output = model(batch_features)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/10:.4f}")

    # Run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(
        model=model,
        sequence_length=60,
        confidence_threshold=0.5,
        cam_threshold=0.5
    )

    result = engine.run(
        data=ohlcv_data,
        initial_capital=100000,
        position_size=0.1,
        transaction_cost=0.001
    )

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nTotal Trades: {len(result.trades)}")
    print(f"Initial Capital: $100,000")
    print(f"Final Capital: ${result.equity_curve[-1]:,.2f}")

    # Show some trade explanations
    if len(result.trades) > 0:
        print("\n" + "=" * 50)
        print("SAMPLE TRADE EXPLANATIONS")
        print("=" * 50)

        for i, trade in enumerate(result.trades[:3]):
            print(f"\nTrade {i+1}:")
            explanation = engine.explain_trade(trade, ohlcv_data)
            for key, value in explanation.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    return result


if __name__ == "__main__":
    result = run_backtest_example()
