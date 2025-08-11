# Grad-CAM for Financial Markets: Visual Explanations from Deep Networks

Gradient-weighted Class Activation Mapping (Grad-CAM) is a powerful interpretability technique that produces visual explanations for decisions made by Convolutional Neural Networks (CNNs). When applied to financial markets, Grad-CAM helps traders and quantitative analysts understand which parts of price charts, candlestick patterns, or technical indicators are most influential in the model's trading decisions.

Deep learning models are often criticized as "black boxes" in finance, where regulatory requirements and risk management demand transparency. Grad-CAM addresses this by highlighting the regions of input data (such as time series patterns or chart images) that contribute most to predictions like "buy," "sell," or "hold" signals.

This chapter covers the theoretical foundations of Grad-CAM, its adaptation for financial time series and chart pattern recognition, and practical implementations in both Python and Rust. We demonstrate applications using both traditional stock market data and cryptocurrency data from the Bybit exchange.

## Content

1. [Introduction to Explainable AI in Finance](#introduction-to-explainable-ai-in-finance)
2. [Understanding Grad-CAM](#understanding-grad-cam)
    * [Mathematical Foundations](#mathematical-foundations)
    * [How Grad-CAM Works](#how-grad-cam-works)
    * [Variants: Grad-CAM++, Score-CAM, and LayerCAM](#variants-grad-cam-score-cam-and-layercam)
3. [CNNs for Financial Data](#cnns-for-financial-data)
    * [Converting Time Series to Images](#converting-time-series-to-images)
    * [Candlestick Chart Recognition](#candlestick-chart-recognition)
    * [Technical Indicator Heatmaps](#technical-indicator-heatmaps)
4. [Implementation](#implementation)
    * [Code Example: Building a CNN for Price Movement Prediction](#code-example-building-a-cnn-for-price-movement-prediction)
    * [Code Example: Applying Grad-CAM to Trading Signals](#code-example-applying-grad-cam-to-trading-signals)
    * [Code Example: Visualizing Important Chart Patterns](#code-example-visualizing-important-chart-patterns)
5. [Backtesting with Interpretable Signals](#backtesting-with-interpretable-signals)
    * [Code Example: Building an Interpretable Trading Strategy](#code-example-building-an-interpretable-trading-strategy)
6. [Rust Implementation](#rust-implementation)
    * [Production-Ready Grad-CAM with Bybit Integration](#production-ready-grad-cam-with-bybit-integration)
7. [References](#references)


## Introduction to Explainable AI in Finance

Explainable Artificial Intelligence (XAI) has become crucial in financial applications for several reasons:

- **Regulatory Compliance**: Financial regulators increasingly require that algorithmic trading decisions be explainable and auditable
- **Risk Management**: Understanding why a model makes certain predictions helps identify potential failure modes
- **Trust Building**: Portfolio managers and clients need to understand the rationale behind AI-driven decisions
- **Model Debugging**: Interpretability tools help identify when models learn spurious correlations

Traditional interpretability methods like feature importance scores work well for tabular data but struggle with the spatial and temporal patterns that CNNs learn from financial charts. Grad-CAM fills this gap by providing intuitive visual explanations that highlight which parts of a chart pattern influenced the model's decision.


## Understanding Grad-CAM

### Mathematical Foundations

Grad-CAM uses the gradient information flowing into the final convolutional layer of a CNN to produce a coarse localization map highlighting the important regions in the input for predicting a target concept.

For a given class $c$, the gradient of the score $y^c$ (before softmax) with respect to feature maps $A^k$ of a convolutional layer is computed as:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

where $Z$ is the number of pixels in the feature map, and $\alpha_k^c$ represents the importance weight of feature map $k$ for class $c$.

The Grad-CAM heatmap is then computed as:

$$L_{Grad-CAM}^c = ReLU\left(\sum_k \alpha_k^c A^k\right)$$

The ReLU is applied because we are only interested in features that have a positive influence on the class of interest. Negative influences likely belong to other classes.

### How Grad-CAM Works

The Grad-CAM algorithm proceeds in the following steps:

1. **Forward Pass**: Pass the input (e.g., a candlestick chart image) through the CNN to obtain the class scores
2. **Backward Pass**: Compute the gradient of the target class score with respect to the feature maps of the last convolutional layer
3. **Weight Computation**: Global average pool the gradients to obtain importance weights for each feature map
4. **Weighted Combination**: Compute a weighted combination of feature maps using these importance weights
5. **ReLU Activation**: Apply ReLU to focus only on positive influences
6. **Upsampling**: Upsample the resulting heatmap to the input image size for visualization

### Variants: Grad-CAM++, Score-CAM, and LayerCAM

Several improvements to the original Grad-CAM have been proposed:

- **Grad-CAM++**: Uses a weighted average of pixel-wise gradients, providing better localization for multiple instances of the same class
- **Score-CAM**: Removes the dependency on gradients entirely, using the forward passing score on each activation map as its weight
- **LayerCAM**: Combines the localization abilities of CAM-based methods with the gradient-free nature of perturbation methods

For financial applications, the original Grad-CAM often suffices, but Grad-CAM++ can be useful when multiple patterns contribute to a single prediction.


## CNNs for Financial Data

### Converting Time Series to Images

Financial time series can be converted to images in several ways for CNN processing:

1. **OHLCV Heatmaps**: Stack multiple time series (Open, High, Low, Close, Volume) as channels
2. **Gramian Angular Fields (GAF)**: Transform time series into polar coordinates and compute angular differences
3. **Markov Transition Fields (MTF)**: Encode transition probabilities between discretized states
4. **Recurrence Plots**: Visualize the recurrence of states in phase space
5. **Candlestick Charts**: Render actual candlestick chart images

Each representation captures different aspects of the data:
- OHLCV heatmaps preserve raw price information
- GAF and MTF capture temporal correlations
- Candlestick charts leverage patterns that traders have used for centuries

### Candlestick Chart Recognition

Candlestick patterns have been used by traders for centuries to predict market movements. Common patterns include:

- **Doji**: Indicates indecision (small body, long wicks)
- **Hammer/Hanging Man**: Reversal signals (small body, long lower shadow)
- **Engulfing Patterns**: Strong reversal signals (large candle engulfs previous)
- **Morning/Evening Star**: Three-candle reversal patterns

CNNs can learn to recognize these patterns directly from chart images, and Grad-CAM reveals which specific patterns the model focuses on.

### Technical Indicator Heatmaps

Multiple technical indicators can be combined into multi-channel images:

- Channel 1: Price relative to moving averages
- Channel 2: RSI (Relative Strength Index)
- Channel 3: MACD histogram
- Channel 4: Bollinger Band position
- Channel 5: Volume relative to average

This representation allows CNNs to learn complex interactions between indicators that would be difficult to capture with traditional feature engineering.


## Implementation

### Code Example: Building a CNN for Price Movement Prediction

The Python implementation in `python/model.py` provides a CNN architecture specifically designed for financial time series classification, along with Grad-CAM functionality:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FinancialCNN(nn.Module):
    """CNN for financial time series classification with Grad-CAM support."""

    def __init__(self, input_channels=5, num_classes=3, sequence_length=60):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

        # For Grad-CAM
        self.gradients = None
        self.activations = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Store activations for Grad-CAM
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        self.activations = x

        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x
```

See the full implementation in `python/model.py` and the training pipeline in `python/train.py`.

### Code Example: Applying Grad-CAM to Trading Signals

The notebook [example.ipynb](python/notebooks/example.ipynb) demonstrates:

1. Loading and preprocessing cryptocurrency data from Bybit
2. Training a CNN to predict price movements
3. Generating Grad-CAM visualizations for trading signals
4. Interpreting which time periods and features drive predictions

### Code Example: Visualizing Important Chart Patterns

The Grad-CAM implementation highlights which parts of the input time series most strongly influenced the model's decision:

```python
class GradCAM:
    """Grad-CAM implementation for 1D financial time series."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.model.gradients
        activations = self.model.activations

        # Weight by global average pooled gradients
        weights = gradients.mean(dim=2, keepdim=True)
        cam = (weights * activations).sum(dim=1)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.squeeze().detach().numpy()
```


## Backtesting with Interpretable Signals

### Code Example: Building an Interpretable Trading Strategy

The backtesting module in `python/backtest.py` implements a trading strategy that:

1. Uses the CNN model to generate buy/sell signals
2. Applies Grad-CAM to explain each trading decision
3. Filters trades based on confidence and interpretability
4. Tracks performance metrics including Sharpe ratio, maximum drawdown, and win rate

Key metrics tracked:
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits divided by gross losses


## Rust Implementation

### Production-Ready Grad-CAM with Bybit Integration

The Rust implementation in `rust/` provides a production-ready system with:

- **High-Performance Inference**: Optimized for low-latency trading
- **Bybit API Integration**: Real-time data fetching and order placement
- **Memory Safety**: Rust's ownership model prevents common bugs
- **Concurrent Processing**: Async runtime for handling multiple symbols

The Rust implementation focuses on efficient inference rather than training, as model training is typically done offline in Python.

See `rust/src/lib.rs` for the main library and `rust/examples/` for usage examples.


## References

1. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**
   - Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
   - URL: https://arxiv.org/abs/1610.02391
   - Year: 2016

2. **Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks**
   - Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V.N.
   - URL: https://arxiv.org/abs/1710.11063
   - Year: 2018

3. **Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks**
   - Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., & Hu, X.
   - URL: https://arxiv.org/abs/1910.01279
   - Year: 2020

4. **Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks**
   - Wang, Z., & Oates, T.
   - AAAI Workshop on Artificial Intelligence for Smart Grids and Buildings
   - Year: 2015

5. **Deep Learning for Financial Applications: A Survey**
   - Ozbayoglu, A.M., Gudelek, M.U., & Sezer, O.B.
   - Applied Soft Computing
   - Year: 2020

6. **Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI**
   - Arrieta, A.B., et al.
   - Information Fusion
   - Year: 2020
