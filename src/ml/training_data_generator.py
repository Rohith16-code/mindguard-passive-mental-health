"""Training data generator module for synthetic data augmentation."""
import numpy as np
import torch
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from scipy.stats import skewnorm, beta
import random


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 1000
    noise_level: float = 0.05
    class_ratio: float = 0.15
    min_samples_per_class: int = 10
    augmentation_factor: float = 2.0


class SyntheticDataGenerator:
    """Generates synthetic data to augment rare event samples."""

    def __init__(self, config: SyntheticConfig = None):
        """Initialize generator with configuration."""
        self.config = config or SyntheticConfig()
        self._feature_stats = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "SyntheticDataGenerator":
        """Fit the generator to real data statistics."""
        self._feature_stats = {
            "mean": np.mean(features, axis=0),
            "std": np.std(features, axis=0) + 1e-8,
            "min": np.min(features, axis=0),
            "max": np.max(features, axis=0),
            "skew": np.mean(features, axis=0),
        }
        self._class_distribution = {
            0: np.sum(labels == 0),
            1: np.sum(labels == 1),
        }
        return self

    def generate(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples to balance and augment the dataset."""
        try:
            self.fit(features, labels)

            # Identify minority class samples
            minority_mask = labels == 1
            majority_mask = labels == 0

            minority_features = features[minority_mask]
            majority_features = features[majority_mask]

            # Calculate how many synthetic samples to generate
            n_minority = len(minority_features)
            n_majority = len(majority_features)

            if n_minority == 0:
                raise ValueError("No positive samples found in training data")

            target_minority = int(n_majority * self.config.class_ratio)
            samples_to_generate = max(
                int(target_minority * self.config.augmentation_factor) - n_minority,
                int(self.config.n_samples * self.config.augmentation_factor),
            )

            # Generate synthetic minority samples using SMOTE-like approach
            synthetic_minority = self._generate_minority_samples(
                minority_features, samples_to_generate
            )

            # Generate synthetic majority samples to maintain class ratio
            synthetic_majority = self._generate_majority_samples(
                majority_features, int(samples_to_generate * 5)
            )

            # Combine real and synthetic data
            synthetic_features = np.vstack(
                [synthetic_minority, synthetic_majority]
            )
            synthetic_labels = np.concatenate(
                [
                    np.ones(len(synthetic_minority)),
                    np.zeros(len(synthetic_majority)),
                ]
            )

            # Shuffle the synthetic data
            indices = np.random.permutation(len(synthetic_features))
            synthetic_features = synthetic_features[indices]
            synthetic_labels = synthetic_labels[indices]

            return synthetic_features, synthetic_labels

        except Exception as e:
            raise RuntimeError(f"Failed to generate synthetic data: {e}")

    def _generate_minority_samples(
        self, real_samples: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """Generate synthetic samples for the minority class."""
        if len(real_samples) == 0:
            raise ValueError("No real minority samples provided")

        synthetic_samples = []
        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(real_samples), 2, replace=False)
            sample1, sample2 = real_samples[idx1], real_samples[idx2]

            # Interpolate with random weight
            alpha = np.random.uniform(0.1, 0.9)
            interpolated = sample1 * alpha + sample2 * (1 - alpha)

            # Add noise
            noise = np.random.normal(
                0, self.config.noise_level * self._feature_stats["std"], len(interpolated)
            )
            synthetic_sample = interpolated + noise

            # Clip to valid range
            synthetic_sample = np.clip(
                synthetic_sample,
                self._feature_stats["min"],
                self._feature_stats["max"],
            )

            synthetic_samples.append(synthetic_sample)

        return np.array(synthetic_samples)

    def _generate_majority_samples(
        self, real_samples: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """Generate synthetic samples for the majority class."""
        if len(real_samples) == 0:
            return np.empty((0, real_samples.shape[1]))

        synthetic_samples = []
        for _ in range(n_samples):
            # Select random sample
            idx = np.random.choice(len(real_samples))
            sample = real_samples[idx]

            # Add noise
            noise = np.random.normal(
                0, self.config.noise_level * self._feature_stats["std"], len(sample)
            )
            synthetic_sample = sample + noise

            # Clip to valid range
            synthetic_sample = np.clip(
                synthetic_sample,
                self._feature_stats["min"],
                self._feature_stats["max"],
            )

            synthetic_samples.append(synthetic_sample)

        return np.array(synthetic_samples)