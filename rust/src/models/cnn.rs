//! CNN model implementation for inference
//!
//! This module provides a simplified CNN implementation for financial time series.
//! Note: This is primarily for demonstration. In production, you would typically
//! load pre-trained weights from a PyTorch model.

use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// CNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNNConfig {
    pub input_channels: usize,
    pub sequence_length: usize,
    pub num_classes: usize,
    pub hidden_dims: Vec<usize>,
}

impl Default for CNNConfig {
    fn default() -> Self {
        Self {
            input_channels: 5,
            sequence_length: 60,
            num_classes: 3,
            hidden_dims: vec![32, 64, 128],
        }
    }
}

/// 1D Convolutional layer
#[derive(Debug, Clone)]
pub struct Conv1d {
    weights: Array3<f64>,  // (out_channels, in_channels, kernel_size)
    bias: Array1<f64>,
    kernel_size: usize,
    padding: usize,
}

impl Conv1d {
    /// Create a new Conv1d layer with random initialization
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, padding: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (in_channels * kernel_size) as f64).sqrt();

        let mut weights = Array3::zeros((out_channels, in_channels, kernel_size));
        for i in 0..out_channels {
            for j in 0..in_channels {
                for k in 0..kernel_size {
                    weights[[i, j, k]] = rng.gen::<f64>() * scale * 2.0 - scale;
                }
            }
        }

        let bias = Array1::zeros(out_channels);

        Self {
            weights,
            bias,
            kernel_size,
            padding,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let (in_channels, seq_len) = input.dim();
        let out_channels = self.weights.dim().0;

        // Compute output length with padding
        let out_len = seq_len + 2 * self.padding - self.kernel_size + 1;
        let mut output = Array2::zeros((out_channels, out_len));

        // Apply convolution
        for oc in 0..out_channels {
            for t in 0..out_len {
                let mut sum = self.bias[oc];
                for ic in 0..in_channels {
                    for k in 0..self.kernel_size {
                        let input_idx = t as isize + k as isize - self.padding as isize;
                        if input_idx >= 0 && (input_idx as usize) < seq_len {
                            sum += input[[ic, input_idx as usize]] * self.weights[[oc, ic, k]];
                        }
                    }
                }
                output[[oc, t]] = sum;
            }
        }

        output
    }

    /// Get output channels
    pub fn out_channels(&self) -> usize {
        self.weights.dim().0
    }
}

/// Linear (fully connected) layer
#[derive(Debug, Clone)]
pub struct Linear {
    weights: Array2<f64>,  // (out_features, in_features)
    bias: Array1<f64>,
}

impl Linear {
    /// Create a new Linear layer with random initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / in_features as f64).sqrt();

        let mut weights = Array2::zeros((out_features, in_features));
        for i in 0..out_features {
            for j in 0..in_features {
                weights[[i, j]] = rng.gen::<f64>() * scale * 2.0 - scale;
            }
        }

        let bias = Array1::zeros(out_features);

        Self { weights, bias }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = self.bias.clone();
        for i in 0..self.weights.dim().0 {
            for j in 0..self.weights.dim().1 {
                output[i] += self.weights[[i, j]] * input[j];
            }
        }
        output
    }
}

/// ReLU activation function
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Apply ReLU to array
pub fn relu_array2(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(relu)
}

/// Global average pooling for 1D data
pub fn global_avg_pool1d(input: &Array2<f64>) -> Array1<f64> {
    let (channels, length) = input.dim();
    let mut output = Array1::zeros(channels);

    for c in 0..channels {
        let mut sum = 0.0;
        for t in 0..length {
            sum += input[[c, t]];
        }
        output[c] = sum / length as f64;
    }

    output
}

/// Softmax function
pub fn softmax(input: &Array1<f64>) -> Array1<f64> {
    let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = input.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    Array1::from(exp_values.iter().map(|x| x / sum).collect::<Vec<f64>>())
}

/// Financial CNN for inference
#[derive(Debug, Clone)]
pub struct FinancialCNN {
    config: CNNConfig,
    conv_layers: Vec<Conv1d>,
    fc1: Linear,
    fc2: Linear,
    // Store activations for Grad-CAM
    last_activations: Option<Array2<f64>>,
}

impl FinancialCNN {
    /// Create a new Financial CNN with random weights
    pub fn new(config: CNNConfig) -> Self {
        let mut conv_layers = Vec::new();
        let mut in_channels = config.input_channels;

        for (i, &out_channels) in config.hidden_dims.iter().enumerate() {
            let kernel_size = if i < config.hidden_dims.len() - 1 { 5 } else { 3 };
            let padding = kernel_size / 2;
            conv_layers.push(Conv1d::new(in_channels, out_channels, kernel_size, padding));
            in_channels = out_channels;
        }

        let last_hidden = *config.hidden_dims.last().unwrap_or(&128);
        let fc1 = Linear::new(last_hidden, last_hidden / 2);
        let fc2 = Linear::new(last_hidden / 2, config.num_classes);

        Self {
            config,
            conv_layers,
            fc1,
            fc2,
            last_activations: None,
        }
    }

    /// Forward pass for a single sample
    ///
    /// Input shape: (channels, sequence_length)
    /// Output shape: (num_classes,)
    pub fn forward(&mut self, input: &Array2<f64>) -> Array1<f64> {
        let mut x = input.clone();

        // Convolutional layers with ReLU
        for conv in &self.conv_layers {
            x = conv.forward(&x);
            x = relu_array2(&x);
        }

        // Store activations for Grad-CAM
        self.last_activations = Some(x.clone());

        // Global average pooling
        let pooled = global_avg_pool1d(&x);

        // Fully connected layers
        let fc1_out = self.fc1.forward(&pooled);
        let fc1_relu = fc1_out.mapv(relu);
        let logits = self.fc2.forward(&fc1_relu);

        logits
    }

    /// Forward pass with batch
    ///
    /// Input shape: (batch_size, channels, sequence_length)
    /// Output shape: (batch_size, num_classes)
    pub fn forward_batch(&mut self, input: &Array3<f64>) -> Array2<f64> {
        let batch_size = input.dim().0;
        let mut outputs = Array2::zeros((batch_size, self.config.num_classes));

        for b in 0..batch_size {
            let sample = input.slice(ndarray::s![b, .., ..]).to_owned();
            let output = self.forward(&sample);
            for c in 0..self.config.num_classes {
                outputs[[b, c]] = output[c];
            }
        }

        outputs
    }

    /// Get prediction (class index) and confidence
    pub fn predict(&mut self, input: &Array2<f64>) -> (usize, f64) {
        let logits = self.forward(input);
        let probs = softmax(&logits);

        let mut max_idx = 0;
        let mut max_prob = probs[0];
        for (i, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                max_idx = i;
            }
        }

        (max_idx, max_prob)
    }

    /// Get the last activations (for Grad-CAM)
    pub fn get_last_activations(&self) -> Option<&Array2<f64>> {
        self.last_activations.as_ref()
    }

    /// Get the configuration
    pub fn config(&self) -> &CNNConfig {
        &self.config
    }

    /// Map class index to label
    pub fn class_to_label(class_idx: usize) -> &'static str {
        match class_idx {
            0 => "Down",
            1 => "Neutral",
            2 => "Up",
            _ => "Unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d() {
        let conv = Conv1d::new(5, 32, 5, 2);
        let input = Array2::zeros((5, 60));
        let output = conv.forward(&input);
        assert_eq!(output.dim(), (32, 60));
    }

    #[test]
    fn test_linear() {
        let linear = Linear::new(128, 3);
        let input = Array1::zeros(128);
        let output = linear.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_financial_cnn() {
        let config = CNNConfig::default();
        let mut model = FinancialCNN::new(config);

        let input = Array2::zeros((5, 60));
        let output = model.forward(&input);
        assert_eq!(output.len(), 3);

        let (pred, conf) = model.predict(&input);
        assert!(pred < 3);
        assert!(conf >= 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_softmax() {
        let input = Array1::from(vec![1.0, 2.0, 3.0]);
        let output = softmax(&input);

        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }
}
