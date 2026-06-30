"""TFLite-compatible LSTM + attention architecture for mental health crisis detection."""
from typing import List, Tuple, Optional
import numpy as np


class AttentionLayer:
    """Custom attention layer compatible with TensorFlow Lite."""

    def __init__(self, units: int = 64):
        self.units = units
        self.W = None
        self.b = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build attention weights."""
        feature_dim = input_shape[-1]
        self.W = np.random.randn(feature_dim, 1).astype(np.float32) * 0.1
        self.b = np.zeros(1, dtype=np.float32)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Compute attention weights and apply to inputs."""
        if self.W is None:
            self.build(inputs.shape)

        # Compute attention scores: score = tanh(W * h + b), shape (B, T, 1)
        scores = np.tanh(np.dot(inputs, self.W) + self.b)  # (B, T, 1)

        # Softmax over time dimension (axis=1)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (B, T, 1)

        # Apply attention weights to inputs: (B, T, F) * (B, T, 1) -> sum over time -> (B, F)
        context = np.sum(attention_weights * inputs, axis=1)  # (B, F)
        return context


class LSTMAttentionLayer:
    """LSTM layer with custom attention mechanism for TFLite compatibility."""

    def __init__(
        self,
        units: int = 64,
        attention_units: int = 64,
        return_sequences: bool = False,
    ):
        self.units = units
        self.attention_units = attention_units
        self.return_sequences = return_sequences
        self.lstm_kernel_i = None
        self.lstm_kernel_f = None
        self.lstm_kernel_c = None
        self.lstm_kernel_o = None
        self.lstm_recurrent_kernel_i = None
        self.lstm_recurrent_kernel_f = None
        self.lstm_recurrent_kernel_c = None
        self.lstm_recurrent_kernel_o = None
        self.lstm_bias_i = None
        self.lstm_bias_f = None
        self.lstm_bias_c = None
        self.lstm_bias_o = None
        self.attention = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build LSTM and attention components."""
        feature_dim = input_shape[-1]

        # Initialize LSTM weights (standard LSTM parameterization)
        def init_weights(input_dim, units):
            return np.random.randn(input_dim, units).astype(np.float32) * 0.1

        def init_recurrent_weights(units):
            return np.random.randn(units, units).astype(np.float32) * 0.1

        def init_bias(units):
            return np.zeros(units, dtype=np.float32)

        self.lstm_kernel_i = init_weights(feature_dim, self.units)
        self.lstm_kernel_f = init_weights(feature_dim, self.units)
        self.lstm_kernel_c = init_weights(feature_dim, self.units)
        self.lstm_kernel_o = init_weights(feature_dim, self.units)

        self.lstm_recurrent_kernel_i = init_recurrent_weights(self.units)
        self.lstm_recurrent_kernel_f = init_recurrent_weights(self.units)
        self.lstm_recurrent_kernel_c = init_recurrent_weights(self.units)
        self.lstm_recurrent_kernel_o = init_recurrent_weights(self.units)

        self.lstm_bias_i = init_bias(self.units)
        self.lstm_bias_f = init_bias(self.units)
        self.lstm_bias_c = init_bias(self.units)
        self.lstm_bias_o = init_bias(self.units)

        # Build attention layer
        self.attention = AttentionLayer(self.attention_units)
        self.attention.build(input_shape[:-1] + (self.units,))

    def _lstm_step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single LSTM step."""
        i_t = np.sigmoid(np.dot(x_t, self.lstm_kernel_i) + np.dot(h_prev, self.lstm_recurrent_kernel_i) + self.lstm_bias_i)
        f_t = np.sigmoid(np.dot(x_t, self.lstm_kernel_f) + np.dot(h_prev, self.lstm_recurrent_kernel_f) + self.lstm_bias_f)
        c_tilde = np.tanh(np.dot(x_t, self.lstm_kernel_c) + np.dot(h_prev, self.lstm_recurrent_kernel_c) + self.lstm_bias_c)
        o_t = np.sigmoid(np.dot(x_t, self.lstm_kernel_o) + np.dot(h_prev, self.lstm_recurrent_kernel_o) + self.lstm_bias_o)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)

        return h_t, c_t

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Process sequence with LSTM and apply attention."""
        if self.lstm_kernel_i is None:
            self.build(inputs.shape)

        batch_size, seq_len, feature_dim = inputs.shape
        h = np.zeros((batch_size, self.units), dtype=np.float32)
        c = np.zeros((batch_size, self.units), dtype=np.float32)

        hidden_states = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h, c = self._lstm_step(x_t, h, c)
            hidden_states.append(h)

        hidden_states = np.stack(hidden_states, axis=1)

        if self.return_sequences:
            return hidden_states

        return self.attention(hidden_states)


def build_lstm_attention_model(
    input_shape: Tuple[int, ...] = (100, 10),
    lstm_units: int = 64,
    attention_units: int = 64,
    output_units: int = 1,
    output_activation: str = "sigmoid",
) -> LSTMAttentionLayer:
    """Build TFLite-compatible LSTM + attention model.

    Args:
        input_shape: Shape of input sequences (seq_len, features)
        lstm_units: Number of LSTM units
        attention_units: Number of attention units
        output_units: Number of output units
        output_activation: Activation for output layer

    Returns:
        LSTMAttentionLayer model instance
    """
    if len(input_shape) != 2:
        raise ValueError("input_shape must be 2D (seq_len, features)")

    model = LSTMAttentionLayer(
        units=lstm_units,
        attention_units=attention_units,
        return_sequences=False,
    )

    # Build with dummy input to initialize weights
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    _ = model(dummy_input)

    return model