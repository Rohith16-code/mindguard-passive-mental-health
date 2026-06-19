"""Hyperparameter tuning module for per-user model calibration."""
from typing import Dict, List, Any, Optional
from itertools import product
from src.ml.models import LogisticRegressionModel
from src.db.client import DBClient
from src.cache.client import RedisClient


class HyperparamTuner:
    """Performs grid search for hyperparameter optimization per user."""

    def __init__(self) -> None:
        """Initialize the hyperparameter tuner."""
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0

    def _build_model(self, params: Dict[str, Any]) -> LogisticRegressionModel:
        """Build and return a model instance with given hyperparameters."""
        return LogisticRegressionModel(
            learning_rate=params.get("lr", 0.01),
            regularization=params.get("reg", 0.001)
        )

    def run_grid_search(
        self,
        data: Dict[str, Any],
        db_client: Optional[DBClient] = None,
        redis_client: Optional[RedisClient] = None
    ) -> Dict[str, Any]:
        """Run grid search over hyperparameter space for a given user's data."""
        user_id = data.get("user_id")
        X = data.get("X", [])
        y = data.get("y", [])
        param_grid = data.get("calibration_params", {})

        if not X or not y:
            raise ValueError("Input data must contain non-empty X and y arrays")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        best_params = None
        best_score = -float("inf")

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            model = self._build_model(params)

            try:
                model.fit(X, y)
                score = model.score(X, y)
            except Exception as e:
                continue

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score

        if user_id and db_client:
            try:
                db_client.save_best_hyperparams(user_id, best_params, best_score)
            except Exception:
                pass

        if user_id and redis_client:
            try:
                redis_client.set_hyperparams_cache(user_id, best_params, best_score)
            except Exception:
                pass

        return {
            "user_id": user_id,
            "best_params": best_params,
            "best_score": best_score
        }


def run_grid_search(
    data: Dict[str, Any],
    db_client: Optional[DBClient] = None,
    redis_client: Optional[RedisClient] = None
) -> Dict[str, Any]:
    """Convenience function to run grid search without instantiating the tuner."""
    tuner = HyperparamTuner()
    return tuner.run_grid_search(data, db_client, redis_client)