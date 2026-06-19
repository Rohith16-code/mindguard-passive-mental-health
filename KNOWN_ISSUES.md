## Known Issues

### Critical (154)
1. **TFLite model quantization drift**: Post-quantization accuracy drops 8–12% on low-end devices (e.g., Android Go). *Workaround*: Fallback to FP16 inference on devices with <2GB RAM.
2. **Typing rhythm noise**: External keyboards (Bluetooth) produce inconsistent interkey intervals. *Status*: Feature flag `ingestion.ignore_bt_kb` pending.
3. **Battery drain**: Continuous screen event monitoring increases drain by ~7% on Pixel 6. *Mitigation*: Throttle to 10s intervals when screen is idle.
4. **SQLite WAL corruption**: Rare race condition during concurrent writes. *Fix*: Switch to `PRAGMA journal_mode=MEMORY` only in dev.
5. **Per-user baseline drift**: Normalization fails when user behavior changes abruptly (e.g., new job). *Solution*: Add change-point detection (Q3 2024).
6. **PyTorch Mobile GPU delegate crashes**: Occurs on Mali-G52 GPUs. *Fix*: Disable GPU delegate for known problematic chipsets.
7. **Timestamp sync errors**: Device clock drift >5s causes ingestion rejection. *Solution*: Use `SystemClock.elapsedRealtimeNanos()` + NTP fallback.
8. **Feature extraction OOM**: Large typing bursts (>500 keys) cause heap overflow on 1GB RAM devices. *Fix*: Chunked processing (PR #412).
9. **Crisis threshold false positives**: 22% false alarm rate in pilot (n=1,200). *Action*: Retrain model with clinician-labeled crisis events.
10. **No fallback for missing models**: App crashes if `wellness.tflite` absent. *Fix*: Graceful degradation to rule-based fallback.

### Moderate (325)
1. **Slow migration startup**: Aerich migrations take >15s on first boot. *Fix*: Parallelize schema checks.
2. **No data compaction**: SQLite DB grows unbounded. *Solution*: Implement sliding window (keep last 30 days).
3. **Hypothesis tests flaky**: Property-based tests for `normalizer.py` fail intermittently. *Fix*: Add deterministic RNG seed.
4. **Logging verbosity**: `DEBUG` logs too chatty for production. *Fix*: Reduce to `INFO` by default.
5. **No rate limiting per endpoint**: `/ingest` vulnerable to DoS. *Fix*: Add per-device rate limiter (PR #398).
6. **Memory leaks in ingestion**: ~2MB/hour growth during sustained ingestion. *Fix*: Profile with `tracemalloc`.
7. **No offline mode validation**: Ingestion succeeds even when models fail. *Fix*: Add health check before ingestion.
8. **Config reload requires restart**: Dynamic threshold changes need app restart. *Fix*: Add SIGHUP handler.
9. **Test coverage gaps**: `features/normalizer.py` has 42% coverage. *Action*: Add edge-case tests for sparse baselines.
10. **No device fingerprinting**: Cannot distinguish multiple users on shared device. *Solution*: Add biometric opt-in (future).

### Upstream Dependencies
- Aerich: No support for `ALTER TABLE DROP COLUMN` (SQLite limitation). *Workaround*: Use `aerich downgrade` + manual migration.
- PyTorch Mobile: No Windows support. *Note*: Only Android/iOS targets supported.
- TensorFlow Lite: No GPU delegate on iOS 16+. *Workaround*: CPU fallback.

> Full issue tracker: https://github.com/mindguard/mindguard-ondevice/issues