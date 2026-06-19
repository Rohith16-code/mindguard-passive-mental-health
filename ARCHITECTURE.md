# Architecture Overview

## Project: mindguard-ondevice

## Pattern
Not specified

## Summary
Passive, privacy-preserving mental health crisis detection using on-device smartphone behavioral signals (typing rhythm, app usage, screen events, response latency) to compute a real-time wellness index. Triggers soft check-ins only when risk exceeds clinician-defined thresholds; all data processed locally with zero raw data transmission.

## Technology Stack
- Python
- FastAPI
- SQLite
- TensorFlow Lite
- PyTorch Mobile
- Aerich
- Hypothesis
- uvloop

## Files to Create
- `[api]` `src/main.py` — entry point
- `[core]` `src/config.py` — runtime config & thresholds
- `[db]` `src/db/models.py` — SQLite schema & ORM models
- `[db]` `src/db/migrations.py` — schema versioning & migrations
- `[ingestion]` `src/ingestion/handler.py` — on-device sensor data ingestion pipeline
- `[ingestion]` `src/ingestion/validator.py` — data sanitization & format checks
- `[features]` `src/features/extractor.py` — feature engineering from raw signals
- `[features]` `src/features/normalizer.py` — per-user baseline calibration
- `[ml]` `src/ml/inference.py` — TFLite model inference + wellness index computation
- `[ml]` `src/ml/training_pipeline.py` — off-device training orchestration
- `[ml]` `src/ml/model_loader.py` — dynamic model versioning & fallback
- `[alerts]` `src/alerts/risk_engine.py` — threshold evaluation + soft-check trigger logic
- `[alerts]` `src/alerts/notifier.py` — local notification dispatch (no external comms)
- `[consent]` `src/consent/manager.py` — user consent tracking & opt-in flows
- `[api]` `src/api/routes.py` — health, status, model update endpoints
- `[api]` `src/api/middleware.py` — on-device rate limiting & anomaly detection
- `[utils]` `src/utils/crypt.py` — local key management & anonymization
- `[utils]` `src/utils/logger.py` — structured on-device logging
- `[workers]` `src/workers/scheduler.py` — periodic analysis & model refresh
- `[workers]` `src/workers/health_monitor.py` — on-device resource & battery-aware scheduling
- `[workers]` `src/workers/anomaly_detector.py` — real-time signal deviation detection
- `[workers]` `src/workers/feedback_processor.py` — user-reported mood feedback ingestion
- `[ml]` `src/ml/data_preprocessor.py` — on-device feature preprocessing
- `[ml]` `src/ml/model_arch.py` — TFLite-compatible LSTM + attention architecture
- `[ml]` `src/ml/hyperparam_tuner.py` — grid search for per-user calibration
- `[db]` `src/db/queries.py` — optimized SQLite query helpers
- `[db]` `src/db/cache.py` — LRU cache for recent features
- `[consent]` `src/consent/protocols.py` — IRB-compliant consent flow definitions
- `[api]` `src/api/schemas.py` — Pydantic models for internal APIs
- `[utils]` `src/utils/time_utils.py` — timezone-aware sleep inference
- `[utils]` `src/utils/metrics.py` — on-device latency & accuracy tracking
- `[workers]` `src/workers/batch_aggregator.py` — sliding-window feature aggregation
- `[ml]` `src/ml/model_validator.py` — model drift detection & rollback
- `[ingestion]` `src/ingestion/sensor_adapter.py` — Android/iOS sensor abstraction layer
- `[alerts]` `src/alerts/crisis_protocol.py` — escalation logic with clinician-defined rules
- `[ml]` `src/ml/federated_aggregator.py` — synthetic model updates from anonymized cohorts
- `[db]` `src/db/metrics_store.py` — performance & usage metrics persistence
- `[workers]` `src/workers/pipeline_orchestrator.py` — async data flow coordination
- `[ml]` `src/ml/explainability.py` — on-device feature attribution for alerts
- `[consent]` `src/consent/anonymizer.py` — k-anonymity preprocessing before local storage
- `[utils]` `src/utils/secure_storage.py` — encrypted local storage for models & keys
- `[api]` `src/api/health_check.py` — liveness/readiness endpoints
- `[workers]` `src/workers/energy_optimizer.py` — CPU/gpu throttling for battery efficiency
- `[ml]` `src/ml/model_compiler.py` — ONNX→TFLite conversion pipeline
- `[ml]` `src/ml/model_registry.py` — versioned model storage & selection
- `[ingestion]` `src/ingestion/buffer.py` — ring buffer for high-frequency signals
- `[ml]` `src/ml/training_data_generator.py` — synthetic data augmentation for rare events
- `[consent]` `src/consent/audit_log.py` — immutable consent & action log

## API Contracts

## Data Models

## Compliance Requirements

## Performance Targets

## Risks

## Infrastructure