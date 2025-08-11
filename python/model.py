"""
Financial CNN with Grad-CAM Support

This module provides a CNN architecture specifically designed for financial time series
classification, along with Grad-CAM implementation for model interpretability.

Classes:
    - FinancialCNN: 1D CNN for OHLCV time series classification
    - FinancialCNN2D: 2D CNN for candlestick chart image classification
    - GradCAM: Grad-CAM implementation for 1D time series
    - GradCAM2D: Grad-CAM implementation for 2D images
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class FinancialCNN(nn.Module):
    """
    1D CNN for financial time series classification with Grad-CAM support.

    This model takes OHLCV (Open, High, Low, Close, Volume) data as multi-channel
    input and predicts price movement direction (down, neutral, up).

    Args:
        input_channels: Number of input channels (default: 5 for OHLCV)
        num_classes: Number of output classes (default: 3 for down/neutral/up)
        sequence_length: Length of input sequence (default: 60 for 60 periods)
        hidden_dims: List of hidden dimensions for conv layers
        dropout: Dropout rate

    Example:
        >>> model = FinancialCNN(input_channels=5, num_classes=3)
        >>> x = torch.randn(32, 5, 60)  # batch of 32, 5 channels, 60 timesteps
        >>> output = model(x)  # shape: (32, 3)
    """

    def __init__(
        self,
        input_channels: int = 5,
        num_classes: int = 3,
        sequence_length: int = 60,
        hidden_dims: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Build convolutional layers
        layers = []
        in_channels = input_channels

        for i, out_channels in enumerate(hidden_dims):
            kernel_size = 5 if i < len(hidden_dims) - 1 else 3
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if i < len(hidden_dims) - 1 else nn.Identity()
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )

        # For Grad-CAM
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        """Hook function to save gradients for Grad-CAM."""
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Feature extraction
        x = self.features(x)

        # Store activations for Grad-CAM
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        self.activations = x

        # Global average pooling
        x = self.gap(x).squeeze(-1)

        # Classification
        x = self.classifier(x)

        return x

    def get_activations_gradient(self) -> Optional[torch.Tensor]:
        """Get the gradients of the activations."""
        return self.gradients

    def get_activations(self) -> Optional[torch.Tensor]:
        """Get the activations of the last convolutional layer."""
        return self.activations


class FinancialCNN2D(nn.Module):
    """
    2D CNN for candlestick chart image classification with Grad-CAM support.

    This model takes rendered candlestick chart images as input and predicts
    price movement direction.

    Args:
        input_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 3)
        pretrained_backbone: Use pretrained ResNet backbone
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 3,
        pretrained_backbone: bool = True
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Simple CNN architecture for chart images
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # For Grad-CAM
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        """Hook function to save gradients for Grad-CAM."""
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)

        # Store activations for Grad-CAM
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        self.activations = x

        x = self.gap(x).flatten(1)
        x = self.classifier(x)

        return x

    def get_activations_gradient(self) -> Optional[torch.Tensor]:
        return self.gradients

    def get_activations(self) -> Optional[torch.Tensor]:
        return self.activations


class GradCAM:
    """
    Grad-CAM implementation for 1D financial time series.

    Produces a coarse localization map highlighting important time periods
    in the input sequence that influenced the model's prediction.

    Args:
        model: The CNN model (must have get_activations and get_activations_gradient methods)

    Example:
        >>> model = FinancialCNN()
        >>> gradcam = GradCAM(model)
        >>> x = torch.randn(1, 5, 60)
        >>> cam = gradcam(x, target_class=2)  # Get CAM for class 2 (up)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input.

        Args:
            input_tensor: Input tensor of shape (1, channels, sequence_length)
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            CAM heatmap as numpy array of shape (sequence_length,)
        """
        # Ensure model is in eval mode but gradients are enabled
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()

        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward pass to get gradients
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not available. "
                             "Ensure forward pass was called with requires_grad=True")

        # Weight activations by global average pooled gradients
        # gradients shape: (batch, channels, length)
        weights = gradients.mean(dim=2, keepdim=True)  # Global average pooling

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)

        # Apply ReLU to focus on positive influences
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    def get_prediction_confidence(self, input_tensor: torch.Tensor) -> Tuple[int, float]:
        """
        Get prediction class and confidence.

        Args:
            input_tensor: Input tensor

        Returns:
            Tuple of (predicted_class, confidence)
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        return pred_class, confidence


class GradCAM2D:
    """
    Grad-CAM implementation for 2D candlestick chart images.

    Produces a heatmap highlighting important regions in the chart image
    that influenced the model's prediction.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_tensor: Input tensor of shape (1, channels, height, width)
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            CAM heatmap as numpy array of shape (height, width)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not available.")

        # Global average pooling of gradients
        weights = gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to input size
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        return cam.squeeze().detach().cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation for improved localization.

    Uses weighted average of pixel-wise gradients for better handling
    of multiple instances of the same pattern.
    """

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not available.")

        # Grad-CAM++ weighting
        # Second derivative approximation
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients

        # Compute alpha coefficients
        sum_activations = activations.sum(dim=2, keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        # Weight by ReLU of gradients
        weights = (alpha * F.relu(gradients)).sum(dim=2, keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()


def create_feature_tensor(
    data: np.ndarray,
    normalize: bool = True
) -> torch.Tensor:
    """
    Create a feature tensor from OHLCV data.

    Args:
        data: OHLCV data array of shape (sequence_length, 5)
              Columns: [Open, High, Low, Close, Volume]
        normalize: Whether to normalize features

    Returns:
        Tensor of shape (1, 5, sequence_length) ready for model input
    """
    # Transpose to (channels, length)
    features = data.T.copy()

    if normalize:
        # Normalize prices relative to first close
        price_scale = features[3, 0]  # First close price
        features[:4] = features[:4] / price_scale - 1  # Relative price change

        # Normalize volume
        vol_mean = features[4].mean()
        vol_std = features[4].std() + 1e-8
        features[4] = (features[4] - vol_mean) / vol_std

    # Convert to tensor and add batch dimension
    tensor = torch.FloatTensor(features).unsqueeze(0)

    return tensor


def visualize_gradcam(
    input_data: np.ndarray,
    cam: np.ndarray,
    prediction: int,
    confidence: float,
    timestamps: Optional[List] = None,
    save_path: Optional[str] = None
):
    """
    Visualize Grad-CAM heatmap overlaid on price data.

    Args:
        input_data: OHLCV data array
        cam: Grad-CAM heatmap
        prediction: Predicted class (0=down, 1=neutral, 2=up)
        confidence: Prediction confidence
        timestamps: Optional list of timestamps
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib is required for visualization. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 1, 1])

    # Price chart with CAM overlay
    ax1 = axes[0]
    close_prices = input_data[:, 3]
    x = np.arange(len(close_prices))

    # Create color map based on CAM values
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = plt.cm.RdYlBu_r(norm(cam))

    # Plot price line segments with CAM colors
    for i in range(len(close_prices) - 1):
        ax1.plot(
            [x[i], x[i + 1]],
            [close_prices[i], close_prices[i + 1]],
            color=colors[i],
            linewidth=2
        )

    ax1.set_ylabel('Price')
    class_names = ['Down', 'Neutral', 'Up']
    ax1.set_title(f'Prediction: {class_names[prediction]} (Confidence: {confidence:.2%})')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Importance')

    # CAM heatmap
    ax2 = axes[1]
    ax2.imshow(cam.reshape(1, -1), aspect='auto', cmap='RdYlBu_r')
    ax2.set_ylabel('Grad-CAM')
    ax2.set_yticks([])

    # Volume with CAM overlay
    ax3 = axes[2]
    volume = input_data[:, 4]
    ax3.bar(x, volume, color=colors, alpha=0.7)
    ax3.set_ylabel('Volume')
    ax3.set_xlabel('Time Period')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing FinancialCNN with Grad-CAM...")

    # Create model
    model = FinancialCNN(input_channels=5, num_classes=3, sequence_length=60)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create sample input
    batch_size = 4
    x = torch.randn(batch_size, 5, 60)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test Grad-CAM
    gradcam = GradCAM(model)
    single_input = torch.randn(1, 5, 60)
    cam = gradcam(single_input)
    print(f"Grad-CAM shape: {cam.shape}")
    print(f"Grad-CAM range: [{cam.min():.3f}, {cam.max():.3f}]")

    # Test Grad-CAM++
    gradcam_pp = GradCAMPlusPlus(model)
    cam_pp = gradcam_pp(single_input)
    print(f"Grad-CAM++ shape: {cam_pp.shape}")

    print("\nAll tests passed!")
