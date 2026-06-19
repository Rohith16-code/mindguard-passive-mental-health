"""On-device feature attribution for mental health crisis detection alerts."""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttributionResult:
    """Feature attribution result for a single prediction."""
    feature_names: List[str]
    attributions: Dict[str, float]
    baseline: Dict[str, float]
    prediction: float
    confidence: float
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())


class SHAPExplainer:
    """SHAP-inspired feature attribution for on-device inference."""

    def __init__(self, model: nn.Module, background_data: Optional[torch.Tensor] = None,
                 n_samples: int = 100, device: str = 'cpu'):
        self.model = model
        self.model.eval()
        self.background_data = background_data
        self.n_samples = n_samples
        self.device = torch.device(device)

    def explain(self, input_tensor: torch.Tensor, feature_names: List[str]) -> AttributionResult:
        """Compute SHAP-like attributions for a single input."""
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)
        baseline = self._compute_baseline(input_tensor)

        attributions = self._compute_shap_values(input_tensor, baseline, feature_names)

        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            prediction = torch.sigmoid(output).item()
            confidence = abs(prediction - 0.5) * 2

        return AttributionResult(
            feature_names=feature_names,
            attributions=attributions,
            baseline={name: float(val) for name, val in zip(feature_names, baseline.squeeze().cpu().numpy())},
            prediction=prediction,
            confidence=confidence
        )

    def _compute_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute baseline (background) for SHAP."""
        if self.background_data is not None:
            return self.background_data.mean(dim=0, keepdim=True)
        else:
            return torch.zeros_like(input_tensor)

    def _compute_shap_values(self, input_tensor: torch.Tensor, baseline: torch.Tensor,
                             feature_names: List[str]) -> Dict[str, float]:
        """Approximate SHAP values using sampling."""
        n_features = input_tensor.shape[-1]
        attributions = torch.zeros(n_features, device=self.device)

        for i in range(n_features):
            # Create perturbed samples
            samples = []
            for _ in range(self.n_samples):
                mask = torch.rand(n_features, device=self.device) > 0.5
                sample = torch.where(mask, input_tensor[0], baseline[0])
                samples.append(sample)
            samples = torch.stack(samples)

            # Evaluate model on samples
            with torch.no_grad():
                outputs = self.model(samples.unsqueeze(1) if samples.dim() == 1 else samples)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions = torch.sigmoid(outputs)

            # Compute marginal contribution for this feature
            contributions = []
            for j in range(self.n_samples):
                mask = torch.rand(n_features, device=self.device) > 0.5
                mask[i] = True
                sample_with = torch.where(mask, input_tensor[0], baseline[0])
                mask[i] = False
                sample_without = torch.where(mask, input_tensor[0], baseline[0])

                with torch.no_grad():
                    pred_with = torch.sigmoid(self.model(sample_with.unsqueeze(0)))
                    pred_without = torch.sigmoid(self.model(sample_without.unsqueeze(0)))

                contributions.append((pred_with - pred_without).item())

            attributions[i] = torch.tensor(contributions).mean()

        # Normalize attributions
        total = attributions.abs().sum()
        if total > 0:
            attributions = attributions / total

        return {name: float(attributions[i].item()) for i, name in enumerate(feature_names)}


class IntegratedGradientsExplainer:
    """Integrated gradients for feature attribution."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.model.eval()
        self.device = torch.device(device)

    def explain(self, input_tensor: torch.Tensor, feature_names: List[str],
                n_steps: int = 50) -> AttributionResult:
        """Compute integrated gradients for a single input."""
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)
        baseline = torch.zeros_like(input_tensor)

        # Compute path integrals
        attributions = torch.zeros_like(input_tensor)

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)

            with torch.enable_grad():
                output = self.model(interpolated)
                if isinstance(output, tuple):
                    output = output[0]
                loss = output.sum()

            grad = torch.autograd.grad(loss, interpolated)[0]
            attributions += grad

        attributions = attributions / n_steps * (input_tensor - baseline)

        # Convert to feature-level attributions
        feature_attribution = attributions.squeeze().cpu().numpy()
        baseline_values = baseline.squeeze().cpu().numpy()

        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            prediction = torch.sigmoid(output).item()
            confidence = abs(prediction - 0.5) * 2

        return AttributionResult(
            feature_names=feature_names,
            attributions={name: float(feature_attribution[i]) for i, name in enumerate(feature_names)},
            baseline={name: float(baseline_values[i]) for i, name in enumerate(feature_names)},
            prediction=prediction,
            confidence=confidence
        )


class LimeExplainer:
    """Lightweight local interpretable model-agnostic explanations."""

    def __init__(self, model: nn.Module, n_perturbations: int = 100,
                 kernel_width: float = 0.25, device: str = 'cpu'):
        self.model = model
        self.model.eval()
        self.n_perturbations = n_perturbations
        self.kernel_width = kernel_width
        self.device = torch.device(device)

    def explain(self, input_tensor: torch.Tensor, feature_names: List[str]) -> AttributionResult:
        """Generate LIME explanations."""
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)
        baseline = torch.zeros_like(input_tensor)

        # Generate perturbed samples
        perturbations = []
        weights = []
        for _ in range(self.n_perturbations):
            noise = torch.randn_like(input_tensor) * 0.1
            sample = input_tensor + noise
            perturbations.append(sample)

            # Compute kernel weight based on distance
            distance = torch.norm(sample - input_tensor).item()
            weight = np.exp(-distance**2 / (2 * self.kernel_width**2))
            weights.append(weight)

        perturbations = torch.cat(perturbations, dim=0).to(self.device)
        weights = torch.tensor(weights, device=self.device)

        # Get predictions for perturbed samples
        with torch.no_grad():
            predictions = torch.sigmoid(self.model(perturbations)).squeeze()

        # Fit simple linear model to approximate local behavior
        # Using pseudo-inverse for efficiency
        A = torch.cat([perturbations, torch.ones((self.n_perturbations, 1), device=self.device)], dim=1)
        b = predictions.unsqueeze(1)

        # Weighted least squares: (A^T W A)^-1 A^T W b
        W = torch.diag(weights)
        try:
            x = torch.linalg.lstsq(A * weights.unsqueeze(1), b * weights).solution
        except Exception:
            x = torch.zeros(A.shape[1], device=self.device)

        coefficients = x[:-1].squeeze().cpu().numpy()

        # Normalize to sum of absolute values = 1
        if np.abs(coefficients).sum() > 0:
            coefficients = coefficients / np.abs(coefficients).sum()

        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            prediction = torch.sigmoid(output).item()
            confidence = abs(prediction - 0.5) * 2

        return AttributionResult(
            feature_names=feature_names,
            attributions={name: float(coefficients[i]) for i, name in enumerate(feature_names)},
            baseline={name: 0.0 for name in feature_names},
            prediction=prediction,
            confidence=confidence
        )


def get_explainer(explainer_type: str = 'shap', **kwargs) -> Any:
    """Factory function to get an explainer instance."""
    if explainer_type.lower() == 'shap':
        return SHAPExplainer(**kwargs)
    elif explainer_type.lower() == 'integrated_gradients':
        return IntegratedGradientsExplainer(**kwargs)
    elif explainer_type.lower() == 'lime':
        return LimeExplainer(**kwargs)
    else:
        raise ValueError(f"Unsupported explainer type: {explainer_type}")