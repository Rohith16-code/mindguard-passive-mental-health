"""ML package — re-exports key components."""

def load_model(path: str):
    """Load ML model from path."""
    from .model_loader import ModelLoader
    loader = ModelLoader()
    return loader.load(path)

def unload_model():
    """Unload current model."""
    pass

__all__ = ["load_model", "unload_model"]
