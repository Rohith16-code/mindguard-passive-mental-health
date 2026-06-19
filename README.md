# MindGuard On-Device

**Privacy-first, on-device mental health crisis detection** using smartphone behavioral signals — no raw data leaves the device.

## Overview

MindGuard-ondevice is a lightweight, real-time wellness monitoring system that passively analyzes user behavior (typing rhythm, app usage, screen events, response latency) to compute a **Wellness Index**. When risk exceeds clinician-defined thresholds, it triggers a *soft check-in* (non-intrusive prompt). All computation happens locally on the device using TensorFlow Lite and PyTorch Mobile models.

### Key Principles
- ✅ **Zero raw data transmission**: Only anonymized features and wellness scores are stored.
- ✅ **Clinician-configurable thresholds**: Risk thresholds and model parameters are editable via config.
- ✅ **On-device inference**: Models run offline with minimal battery/CPU overhead.
- ✅ **Modular & testable**: Clean separation of ingestion, feature engineering, and inference.

## Architecture

```
[Smartphone Sensors] → [Ingestion Pipeline] → [Feature Extraction & Normalization]
                              ↓
                    [Wellness Index Computation (TFLite/PyTorch Mobile)]
                              ↓
                    [SQLite Storage (anonymized features + scores)]
                              ↓
                    [FastAPI Health/Config Endpoints]
```

## Tech Stack
- Python 3.10+
- FastAPI (async API server)
- SQLite + Aerich (schema migrations)
- TensorFlow Lite & PyTorch Mobile (on-device models)
- Hypothesis (property-based testing)
- uvloop (high-performance event loop)

## Quick Start

```bash
# Clone & install
pip install -r requirements.txt

# Initialize DB & run migrations
python -m src.db.migrations upgrade head

# Start server (with uvloop)
python -m src.main
```

⚠️ **Note**: Model files (`model.tflite`, `model.pt`) must be placed in `assets/models/` before inference can run.

## Configuration
Edit `config.yaml` (or set env vars) to adjust:
- `thresholds.crisis_risk`: Risk score triggering soft check-in
- `ingestion.sample_rate`: Sensor sampling frequency (Hz)
- `features.normalization_window`: Baseline calibration period (days)
- `model.device`: `cpu` or `gpu` (for PyTorch Mobile)

## Privacy & Compliance
- All data remains on-device; no telemetry, no cloud.
- Anonymized feature vectors are stored with device IDs hashed via SHA-256.
- Designed for HIPAA/GDPR alignment (consult legal before deployment).

---
*For clinicians: Thresholds and model weights are adjustable per patient profile.*