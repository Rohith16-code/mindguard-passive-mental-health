"""Inference module for mental health crisis detection using TFLite models."""
import os
import json
from typing import List, Union, Dict, Any

import numpy as np
import tensorflow as tf

from src.config import settings
from src.db.session import SessionLocal
from src.db.models import UserBehavior, WellnessLog


def load_model(model_path: str) -> tf.lite.Interpreter:
    """Load a TFLite model from the given path.

    Args:
        model_path: Path to the .tflite model file.

    Returns:
        A tf.lite.Interpreter instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter: tf.lite.Interpreter, input_data: Union[List[float], np.ndarray]) -> float:
    """Run inference using the provided TFLite interpreter.

    Args:
        interpreter: A tf.lite.Interpreter instance.
        input_data: Input features as a list or numpy array.

    Returns:
        A float between 0.0 and 1.0 representing crisis probability.

    Raises:
        ValueError: If input shape does not match model expectations.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if isinstance(input_data, list):
        input_data = np.array(input_data, dtype=np.float32)

    input_shape = input_details[0]['shape']
    expected_shape = tuple(input_shape)

    if input_data.shape != expected_shape:
        raise ValueError(
            f"Input shape {input_data.shape} does not match expected shape {expected_shape}"
        )

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = float(output_data.flatten()[0])

    return max(0.0, min(1.0, result))


def compute_wellness_index(user_id: int, crisis_probability: float) -> float:
    """Compute a wellness index from crisis probability and historical behavior.

    Args:
        user_id: ID of the user.
        crisis_probability: Crisis probability from model inference (0.0 to 1.0).

    Returns:
        Wellness index (0.0 to 100.0), where higher is better.
    """
    db = SessionLocal()
    try:
        recent_behaviors = (
            db.query(UserBehavior)
            .filter(UserBehavior.user_id == user_id)
            .order_by(UserBehavior.timestamp.desc())
            .limit(7)
            .all()
        )

        if not recent_behaviors:
            base_wellness = 50.0
        else:
            avg_sleep_hours = np.mean([b.sleep_hours for b in recent_behaviors if b.sleep_hours is not None])
            avg_activity = np.mean([b.physical_activity for b in recent_behaviors if b.physical_activity is not None])
            avg_social = np.mean([b.social_interaction for b in recent_behaviors if b.social_interaction is not None])
            avg_journal_sentiment = np.mean([b.journal_sentiment for b in recent_behaviors if b.journal_sentiment is not None])

            weights = {
                'sleep': 0.25,
                'activity': 0.20,
                'social': 0.25,
                'journal': 0.30,
            }

            normalized_sleep = min(avg_sleep_hours / 8.0, 1.0) * 100.0
            normalized_activity = min(avg_activity / 10.0, 1.0) * 100.0
            normalized_social = min(avg_social / 10.0, 1.0) * 100.0
            normalized_journal = (avg_journal_sentiment + 1.0) / 2.0 * 100.0

            base_wellness = (
                weights['sleep'] * normalized_sleep +
                weights['activity'] * normalized_activity +
                weights['social'] * normalized_social +
                weights['journal'] * normalized_journal
            )
        wellness_score = base_wellness * (1.0 - crisis_probability) + 25.0 * crisis_probability
        return max(0.0, min(100.0, round(wellness_score, 2)))

    finally:
        db.close()