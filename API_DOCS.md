## API Documentation

### Endpoints

#### `POST /ingest`
Ingest raw behavioral signals.

**Request Body** (JSON):
```json
{
  "device_id_hash": "sha256:abc123...",
  "timestamp": "2024-05-20T14:30:00Z",
  "signals": {
    "typing_rhythm": {"interkey_ms": [120, 95, 110, ...]},
    "app_usage": [{"app": "com.example.chat", "duration_s": 45}],
    "screen_events": [{"type": "unlock", "timestamp": "..."}],
    "response_latency_ms": 850
  }
}
```

**Responses**:
- `200 OK`: `{"status": "ingested", "wellness_index": 0.62}`
- `400`: Invalid data (see validator errors)
- `429`: Rate-limited (max 100 events/min/device)

---

#### `GET /health`
System health & model status.

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": ["wellness.tflite"],
  "db_version": "v3.2.1",
  "last_wellness_update": "2024-05-20T14:30:00Z"
}
```

---

#### `GET /config`
Get current thresholds & config (admin-only).

**Auth**: `X-API-Key: <admin-key>`

**Response**:
```json
{
  "thresholds": {
    "crisis_risk": 0.75,
    "warning_risk": 0.5
  },
  "features": {
    "normalization_window": 7
  }
}
```

---

#### `PUT /config/thresholds`
Update thresholds (admin-only).

**Request Body**:
```json
{
  "crisis_risk": 0.8,
  "warning_risk": 0.55
}
```

**Responses**:
- `200 OK`: Updated config
- `403`: Unauthorized
- `422`: Invalid threshold values

---

#### `GET /wellness/latest`
Get latest wellness index for a device.

**Query**: `?device_id_hash=sha256:abc123...`

**Response**:
```json
{
  "device_id_hash": "sha256:abc123...",
  "timestamp": "2024-05-20T14:30:00Z",
  "wellness_index": 0.62,
  "risk_level": "moderate"
}
```

---

### Data Models

#### Signal Types
| Field | Type | Description |
|-------|------|-------------|
| `interkey_ms` | `List[int]` | Time between keypresses (ms) |
| `app_usage` | `List[AppUsage]` | App session events |
| `screen_events` | `List[ScreenEvent]` | Lock/unlock/battery events |
| `response_latency_ms` | `int` | Time to respond to a prompt |

#### Wellness Index
- Range: `0.0` (high risk) to `1.0` (optimal wellness)
- Computed via ensemble of TFLite/PyTorch models + per-user normalization.
- Risk levels: `low` (<0.4), `moderate` (0.4–0.7), `high` (>0.7)

### Error Codes
| Code | Meaning |
|------|---------|
| `INGEST-001` | Missing required signal field |
| `INGEST-002` | Timestamp too old (>24h) |
| `MODEL-001` | Inference failed (model not loaded) |
| `DB-001` | Migration out of sync |
| `RATE-001` | Rate limit exceeded |

> Full error codes in `src/errors.py`