## Setup & Deployment

### Prerequisites
- Python 3.10+
- Android SDK (for PyTorch Mobile) *or* Android NDK (for TFLite GPU delegates)
- SQLite 3.35+ (built-in on modern OSes)

### Installation
```bash
# Clone repo
git clone https://github.com/mindguard/mindguard-ondevice.git
cd mindguard-ondevice

# Create virtual env
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m src.db.migrations init
python -m src.db.migrations migrate
python -m src.db.migrations upgrade head

# Place models (example)
mkdir -p assets/models
cp ../model.tflite assets/models/wellness.tflite
```

### Running
```bash
# Development (with auto-reload)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production (uvloop + gunicorn)
gunicorn src.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Testing
```bash
# Run all tests (Hypothesis + pytest)
pytest tests/ -v --hypothesis-show-statistics

# Property-based tests for feature normalizer
pytest tests/features/test_normalizer.py --hypothesis
```

### Configuration
Create `config.yaml` (or use defaults):
```yaml
database:
  url: "sqlite:///data/mindguard.db"

thresholds:
  crisis_risk: 0.75
  warning_risk: 0.5

features:
  normalization_window: 7  # days
  max_samples_per_user: 10000

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Android Integration
1. Add `mindguard-ondevice` as a Gradle module.
2. Use `PyTorchMobile`/`TFLite` delegates in `onDeviceInference.kt`.
3. Send sensor events via `IngestionService` (Kotlin) → HTTP POST to local FastAPI.

See `examples/android/` for sample integration.