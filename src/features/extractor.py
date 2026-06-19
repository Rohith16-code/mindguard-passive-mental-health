"""Feature extraction module for signal processing."""
import math
from typing import List, Dict, Any, Optional


def normalize_signal(signal: List[float]) -> List[float]:
    """Normalize a signal to [0, 1] range based on min-max scaling.
    
    Args:
        signal: List of numeric values.
        
    Returns:
        Normalized signal where min value becomes 0.0 and max becomes 1.0.
    """
    if not signal:
        return []
    
    min_val = min(signal)
    max_val = max(signal)
    
    if min_val == max_val:
        return [0.0] * len(signal)
    
    range_val = max_val - min_val
    return [(x - min_val) / range_val for x in signal]


def extract_features(signal: List[float]) -> Dict[str, Optional[float]]:
    """Extract statistical features from a signal.
    
    Args:
        signal: List of numeric values representing a time series signal.
        
    Returns:
        Dictionary containing statistical features: mean, std, min, max, sum.
    """
    if not signal:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "sum": 0.0
        }
    
    n = len(signal)
    total = sum(signal)
    mean = total / n
    
    if n == 1:
        std = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in signal) / n
        std = math.sqrt(variance)
    
    return {
        "mean": mean,
        "std": std,
        "min": min(signal),
        "max": max(signal),
        "sum": total
    }


class SignalExtractor:
    """Extractor class for processing and extracting features from signals."""
    
    def __init__(self) -> None:
        """Initialize the SignalExtractor."""
        pass
    
    def extract(self, signal: List[float]) -> Dict[str, Any]:
        """Extract features from a signal.
        
        Args:
            signal: List of numeric values representing a time series signal.
            
        Returns:
            Dictionary containing extracted features and metadata.
        """
        features = extract_features(signal)
        normalized = normalize_signal(signal)
        
        return {
            "features": features,
            "normalized": normalized,
            "length": len(signal)
        }