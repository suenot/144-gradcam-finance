//! Grad-CAM implementation for model interpretability
//!
//! This module provides Grad-CAM (Gradient-weighted Class Activation Mapping)
//! for interpreting CNN predictions on financial time series.
//!
//! Note: This is a simplified implementation using numerical differentiation
//! since Rust doesn't have automatic differentiation built-in. For production,
//! consider using a framework like tch-rs (PyTorch bindings) or candle.

use crate::models::cnn::FinancialCNN;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Result of Grad-CAM computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradCAMResult {
    /// CAM heatmap values (length = sequence_length)
    pub heatmap: Vec<f64>,
    /// Predicted class index
    pub predicted_class: usize,
    /// Prediction confidence
    pub confidence: f64,
    /// Important time periods (indices where heatmap > threshold)
    pub important_periods: Vec<usize>,
    /// Focus statistics
    pub focus_stats: FocusStats,
}

/// Statistics about model focus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusStats {
    /// Average attention on recent periods (last 10)
    pub recent_focus: f64,
    /// Average attention on historical periods (first 10)
    pub historical_focus: f64,
    /// Ratio of recent to historical focus
    pub focus_ratio: f64,
    /// Period with maximum attention
    pub peak_period: usize,
    /// Maximum attention value
    pub peak_value: f64,
}

/// Grad-CAM implementation for 1D time series
pub struct GradCAM {
    epsilon: f64,
    cam_threshold: f64,
}

impl Default for GradCAM {
    fn default() -> Self {
        Self::new(1e-4, 0.5)
    }
}

impl GradCAM {
    /// Create a new Grad-CAM instance
    ///
    /// # Arguments
    /// * `epsilon` - Small value for numerical differentiation
    /// * `cam_threshold` - Threshold for identifying important periods
    pub fn new(epsilon: f64, cam_threshold: f64) -> Self {
        Self {
            epsilon,
            cam_threshold,
        }
    }

    /// Compute Grad-CAM heatmap for an input
    ///
    /// This uses a simplified approach based on activation magnitudes and
    /// numerical gradients, as full automatic differentiation is not available.
    ///
    /// # Arguments
    /// * `model` - The CNN model
    /// * `input` - Input array of shape (channels, sequence_length)
    /// * `target_class` - Optional target class. If None, uses predicted class.
    pub fn compute(
        &self,
        model: &mut FinancialCNN,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> GradCAMResult {
        let (channels, seq_len) = input.dim();

        // Get base prediction
        let (pred_class, confidence) = model.predict(input);
        let target = target_class.unwrap_or(pred_class);

        // Get activations from the last convolutional layer
        let activations = model.get_last_activations().cloned();

        // Compute importance scores using perturbation-based method
        let mut importance = vec![0.0; seq_len];

        // For each time step, compute how much perturbing it affects the target class
        for t in 0..seq_len {
            let mut perturbed = input.clone();

            // Zero out this time step across all channels
            for c in 0..channels {
                perturbed[[c, t]] = 0.0;
            }

            // Get prediction with perturbation
            let perturbed_logits = model.forward(&perturbed);

            // Compute original logits
            let original_logits = model.forward(input);

            // Importance = drop in target class score when this time step is masked
            importance[t] = (original_logits[target] - perturbed_logits[target]).max(0.0);
        }

        // If we have activations, combine with activation magnitudes
        if let Some(acts) = activations {
            let (act_channels, act_len) = acts.dim();

            // Compute channel-wise importance from activations
            let mut act_importance = vec![0.0; act_len];
            for t in 0..act_len {
                for c in 0..act_channels {
                    act_importance[t] += acts[[c, t]].abs();
                }
                act_importance[t] /= act_channels as f64;
            }

            // Combine perturbation importance with activation importance
            // Upsample activation importance if needed
            if act_len != seq_len {
                let upsampled = upsample(&act_importance, seq_len);
                for t in 0..seq_len {
                    importance[t] = importance[t] * 0.5 + upsampled[t] * 0.5;
                }
            } else {
                for t in 0..seq_len {
                    importance[t] = importance[t] * 0.5 + act_importance[t] * 0.5;
                }
            }
        }

        // Apply ReLU (keep only positive importance)
        for val in &mut importance {
            *val = val.max(0.0);
        }

        // Normalize to [0, 1]
        let max_val = importance.iter().cloned().fold(0.0_f64, f64::max);
        let min_val = importance.iter().cloned().fold(f64::MAX, f64::min);
        let range = max_val - min_val;

        let heatmap: Vec<f64> = if range > 1e-8 {
            importance.iter().map(|x| (x - min_val) / range).collect()
        } else {
            vec![0.5; seq_len]
        };

        // Find important periods
        let important_periods: Vec<usize> = heatmap
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > self.cam_threshold)
            .map(|(i, _)| i)
            .collect();

        // Compute focus statistics
        let focus_stats = self.compute_focus_stats(&heatmap);

        GradCAMResult {
            heatmap,
            predicted_class: pred_class,
            confidence,
            important_periods,
            focus_stats,
        }
    }

    /// Compute focus statistics from heatmap
    fn compute_focus_stats(&self, heatmap: &[f64]) -> FocusStats {
        let len = heatmap.len();
        let recent_start = len.saturating_sub(10);

        let recent_focus = if recent_start < len {
            heatmap[recent_start..].iter().sum::<f64>() / (len - recent_start) as f64
        } else {
            0.0
        };

        let historical_focus = if len >= 10 {
            heatmap[..10].iter().sum::<f64>() / 10.0
        } else {
            heatmap.iter().sum::<f64>() / len as f64
        };

        let focus_ratio = if historical_focus > 1e-8 {
            recent_focus / historical_focus
        } else if recent_focus > 1e-8 {
            f64::INFINITY
        } else {
            1.0
        };

        let (peak_period, peak_value) = heatmap
            .iter()
            .enumerate()
            .fold((0, 0.0), |(max_idx, max_val), (idx, &val)| {
                if val > max_val {
                    (idx, val)
                } else {
                    (max_idx, max_val)
                }
            });

        FocusStats {
            recent_focus,
            historical_focus,
            focus_ratio,
            peak_period,
            peak_value,
        }
    }

    /// Generate text explanation for a Grad-CAM result
    pub fn explain(&self, result: &GradCAMResult) -> String {
        let class_name = FinancialCNN::class_to_label(result.predicted_class);
        let mut explanation = format!(
            "Prediction: {} (confidence: {:.1}%)\n",
            class_name,
            result.confidence * 100.0
        );

        explanation.push_str(&format!(
            "Peak attention at period {} (value: {:.3})\n",
            result.focus_stats.peak_period, result.focus_stats.peak_value
        ));

        if result.focus_stats.focus_ratio > 1.5 {
            explanation.push_str("Model focuses primarily on RECENT price action.\n");
        } else if result.focus_stats.focus_ratio < 0.67 {
            explanation.push_str("Model focuses primarily on HISTORICAL price action.\n");
        } else {
            explanation.push_str("Model attention is balanced across the time window.\n");
        }

        explanation.push_str(&format!(
            "Important periods: {:?}\n",
            result.important_periods
        ));

        explanation
    }
}

/// Upsample a 1D array to a target length using linear interpolation
fn upsample(input: &[f64], target_len: usize) -> Vec<f64> {
    if input.is_empty() {
        return vec![0.0; target_len];
    }

    let mut output = Vec::with_capacity(target_len);
    let scale = (input.len() - 1) as f64 / (target_len - 1).max(1) as f64;

    for i in 0..target_len {
        let src_idx = i as f64 * scale;
        let low = src_idx.floor() as usize;
        let high = (low + 1).min(input.len() - 1);
        let frac = src_idx - low as f64;

        let value = input[low] * (1.0 - frac) + input[high] * frac;
        output.push(value);
    }

    output
}

/// Smooth a heatmap using a moving average
pub fn smooth_heatmap(heatmap: &[f64], window_size: usize) -> Vec<f64> {
    if heatmap.len() < window_size {
        return heatmap.to_vec();
    }

    let mut smoothed = Vec::with_capacity(heatmap.len());
    let half_window = window_size / 2;

    for i in 0..heatmap.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(heatmap.len());
        let sum: f64 = heatmap[start..end].iter().sum();
        smoothed.push(sum / (end - start) as f64);
    }

    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cnn::CNNConfig;

    #[test]
    fn test_gradcam_compute() {
        let config = CNNConfig::default();
        let mut model = FinancialCNN::new(config);
        let gradcam = GradCAM::default();

        let input = Array2::zeros((5, 60));
        let result = gradcam.compute(&mut model, &input, None);

        assert_eq!(result.heatmap.len(), 60);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.predicted_class < 3);
    }

    #[test]
    fn test_upsample() {
        let input = vec![0.0, 1.0, 0.0];
        let output = upsample(&input, 5);
        assert_eq!(output.len(), 5);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
        assert!((output[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_heatmap() {
        let heatmap = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let smoothed = smooth_heatmap(&heatmap, 3);
        assert_eq!(smoothed.len(), 5);
        assert!(smoothed[2] > smoothed[0]);
        assert!(smoothed[1] > 0.0);
    }

    #[test]
    fn test_focus_stats() {
        let gradcam = GradCAM::default();
        let heatmap: Vec<f64> = (0..60).map(|i| i as f64 / 60.0).collect();
        let stats = gradcam.compute_focus_stats(&heatmap);

        assert!(stats.recent_focus > stats.historical_focus);
        assert!(stats.focus_ratio > 1.0);
        assert_eq!(stats.peak_period, 59);
    }
}
