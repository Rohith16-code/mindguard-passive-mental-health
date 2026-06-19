"""Training pipeline orchestration module for off-device model training."""
import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from aerich import Command
from aerich.tortoise import Tortoise
from fastapi import HTTPException
from tortoise import Tortoise as TortoiseORM
from tortoise.exceptions import ConfigurationError, DBConnectionError

from src.config import settings
from src.ingestion.buffer import RingBuffer
from src.ml.data_preprocessor import Preprocessor
from src.ml.model_compiler import ModelCompiler
from src.ml.model_registry import ModelRegistry
from src.ml.model_validator import ModelValidator
from src.ml.training_data_generator import SyntheticDataGenerator
from src.utils.metrics import MetricsTracker
from src.utils.time_utils import get_utc_now

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates off-device training pipeline with data ingestion, augmentation, training, and deployment."""

    def __init__(
        self,
        buffer: RingBuffer,
        preprocessor: Preprocessor,
        synthetic_generator: SyntheticDataGenerator,
        model_compiler: ModelCompiler,
        model_registry: ModelRegistry,
        model_validator: ModelValidator,
        metrics_tracker: MetricsTracker,
    ):
        self.buffer = buffer
        self.preprocessor = preprocessor
        self.synthetic_generator = synthetic_generator
        self.model_compiler = model_compiler
        self.model_registry = model_registry
        self.model_validator = model_validator
        self.metrics_tracker = metrics_tracker
        self._is_running = False
        self._training_lock = asyncio.Lock()
        self._last_training_time: Optional[datetime] = None

    async def initialize(self) -> None:
        """Initialize database and model registry."""
        try:
            await Tortoise.init(
                db_url=settings.DATABASE_URL,
                modules={"models": ["src.models"]},
            )
            await Tortoise.generate_schemas()
            logger.info("Tortoise ORM initialized successfully")
        except (ConfigurationError, DBConnectionError) as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up database connections."""
        await Tortoise.close_connections()
        logger.info("Tortoise ORM connections closed")

    async def collect_training_data(
        self, window_hours: int = 24, min_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect and preprocess training data from ring buffer."""
        try:
            now = get_utc_now()
            cutoff = now - timedelta(hours=window_hours)

            raw_data = self.buffer.get_since(cutoff)
            if len(raw_data) < min_samples:
                logger.warning(
                    f"Insufficient data: {len(raw_data)} < {min_samples}"
                )
                raise ValueError("Insufficient training data")

            features, labels = self.preprocessor.process(raw_data)
            logger.info(
                f"Collected {len(features)} samples with shape {features.shape}"
            )
            return features, labels
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            raise

    async def generate_synthetic_data(
        self, real_features: np.ndarray, real_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data to augment real training data."""
        try:
            synthetic_features, synthetic_labels = self.synthetic_generator.generate(
                real_features, real_labels
            )
            logger.info(
                f"Generated {len(synthetic_features)} synthetic samples"
            )
            return synthetic_features, synthetic_labels
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise

    async def train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.15,
    ) -> Dict[str, Any]:
        """Train model on augmented data."""
        try:
            torch.manual_seed(settings.TRAINING_SEED)
            np.random.seed(settings.TRAINING_SEED)

            # Convert to tensors
            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32)

            # Split data
            n = len(X)
            indices = torch.randperm(n)
            split = int(n * (1 - validation_split))
            train_idx, val_idx = indices[:split], indices[split:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Initialize model
            input_dim = X_train.shape[1]
            model = self._build_model(input_dim)

            # Setup optimizer and loss
            optimizer = torch.optim.Adam(
                model.parameters(), lr=settings.LEARNING_RATE
            )
            criterion = torch.nn.BCEWithLogitsLoss()

            # Training loop
            best_val_loss = float("inf")
            best_model_state = None

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                # Forward pass
                outputs = model(X_train)
                loss = criterion(outputs, y_train)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()

                # Track metrics
                self.metrics_tracker.log_metric(
                    "train_loss", loss.item(), step=epoch
                )
                self.metrics_tracker.log_metric(
                    "val_loss", val_loss, step=epoch
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}"
                    )

            # Load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Final evaluation
            model.eval()
            with torch.no_grad():
                train_pred = torch.sigmoid(model(X_train))
                val_pred = torch.sigmoid(model(X_val))

            metrics = {
                "train_loss": loss.item(),
                "val_loss": best_val_loss,
                "train_accuracy": self._compute_accuracy(
                    train_pred, y_train
                ),
                "val_accuracy": self._compute_accuracy(val_pred, y_val),
            }

            logger.info(f"Training completed: {metrics}")
            return {"model": model, "metrics": metrics}

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def _build_model(self, input_dim: int) -> torch.nn.Module:
        """Build model architecture."""
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
        )

    def _compute_accuracy(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Compute binary accuracy."""
        preds = (predictions > 0.5).float()
        return (preds == targets).float().mean().item()

    async def compile_and_register_model(
        self, model: torch.nn.Module, metrics: Dict[str, Any]
    ) -> str:
        """Compile model to TFLite and register in model registry."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.pt"
                torch.save(model.state_dict(), model_path)

                # Compile to TFLite
                tflite_path = await self.model_compiler.compile(
                    model_path, output_dir=tmpdir
                )

                # Validate model
                validation_result = await self.model_validator.validate(
                    tflite_path
                )

                # Register model
                model_version = await self.model_registry.register(
                    model_path=str(tflite_path),
                    metrics=metrics,
                    validation_result=validation_result,
                    metadata={
                        "training_time": get_utc_now().isoformat(),
                        "validation_status": validation_result.status,
                    },
                )

                logger.info(f"Model registered as version {model_version}")
                return model_version

        except Exception as e:
            logger.error(f"Error in model compilation and registration: {e}")
            raise

    async def run_training_cycle(self) -> Optional[str]:
        """Execute full training pipeline."""
        async with self._training_lock:
            try:
                logger.info("Starting training cycle")

                # Collect real data
                features, labels = await self.collect_training_data()

                # Generate synthetic data
                synth_features, synth_labels = await self.generate_synthetic_data(
                    features, labels
                )

                # Combine datasets
                combined_features = np.vstack([features, synth_features])
                combined_labels = np.concatenate([labels, synth_labels])

                # Train model
                result = await self.train_model(
                    combined_features, combined_labels
                )

                # Compile and register
                model_version = await self.compile_and_register_model(
                    result["model"], result["metrics"]
                )

                self._last_training_time = get_utc_now()
                logger.info(f"Training cycle completed: version {model_version}")
                return model_version

            except Exception as e:
                logger.error(f"Training cycle failed: {e}")
                raise

    async def schedule_training(self, interval_hours: int = 24) -> None:
        """Schedule periodic training cycles."""
        self._is_running = True
        while self._is_running:
            try:
                next_run = (
                    self._last_training_time
                    or get_utc_now() - timedelta(hours=interval_hours)
                ) + timedelta(hours=interval_hours)

                wait_time = (next_run - get_utc_now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                await self.run_training_cycle()

            except asyncio.CancelledError:
                logger.info("Training scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in training scheduler: {e}")
                await asyncio.sleep(60)

    def stop(self) -> None:
        """Stop the training scheduler."""
        self._is_running = False


async def run_training_pipeline() -> None:
    """Entry point for training pipeline execution."""
    pipeline = TrainingPipeline(
        buffer=RingBuffer(max_size=settings.BUFFER_SIZE),
        preprocessor=Preprocessor(),
        synthetic_generator=SyntheticDataGenerator(),
        model_compiler=ModelCompiler(),
        model_registry=ModelRegistry(),
        model_validator=ModelValidator(),
        metrics_tracker=MetricsTracker(),
    )

    try:
        await pipeline.initialize()
        await pipeline.schedule_training(interval_hours=24)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise
    finally:
        await pipeline.cleanup()