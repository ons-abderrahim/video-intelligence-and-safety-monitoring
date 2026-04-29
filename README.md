# video-intelligence-and-safety-monitoring
Real-time behavior analysis from video streams for workplace or retail environments


# 🎥 VISP — Video Intelligence & Safety Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Real-time behavior analysis from video streams for workplace and retail environments.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Docs](#-api-reference) · [Deployment](#-deployment)

</div>

---

## 🧠 What is VISP?

VISP is an AI-powered safety monitoring platform that ingests live or recorded video streams and detects safety-critical events in real time — including **violence**, **PPE non-compliance**, and **restricted area breaches** — without requiring custom hardware.

Built on top of state-of-the-art video transformers (MViT / ViViT / R(2+1)D) and optimized for edge deployment via ONNX, VISP bridges the gap between cutting-edge research and production-ready enterprise tooling.

> 💡 **Target markets**: Manufacturing, construction, retail loss prevention, warehouse logistics, healthcare compliance.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔴 **Real-time Detection** | WebSocket stream with per-frame event confidence scores |
| 🧠 **Multi-model Support** | MViT-v2, ViViT-B, R(2+1)D — swap at runtime via config |
| 🏭 **PPE Compliance** | Hard hat, vest, gloves, and mask detection |
| 🚧 **Zone Intrusion** | Polygon-based restricted area monitoring |
| ⚠️ **Violence Detection** | Behavior classification with temporal context |
| 📊 **Live Dashboard** | React UI with event log, heatmaps, and alert timeline |
| 📦 **ONNX Export** | Optimized edge deployment — runs on CPU / Jetson |
| 🔔 **Alert System** | Webhooks, email, and Slack notifications |
| 🗂️ **Event Archive** | Redis-backed event queue + PostgreSQL persistence |
| 🐳 **Docker-ready** | Single `docker compose up` for full stack |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Video Sources                       │
│         RTSP Cameras │ HTTP Streams │ File Upload        │
└────────────────────────────┬────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Ingest Layer   │
                    │  (FastAPI + WS)  │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │      Inference Engine        │
              │  MViT / ViViT / R(2+1)D     │
              │     (PyTorch + ONNX RT)      │
              └──────┬──────────────┬────────┘
                     │              │
           ┌─────────▼──┐    ┌──────▼──────────┐
           │ Event Queue │    │  Alert Service   │
           │  (Redis)    │    │ (Webhook/Email)  │
           └─────────┬───┘    └─────────────────┘
                     │
           ┌─────────▼──────────┐
           │    React Dashboard  │
           │  Live feed · Logs  │
           │  Heatmap · Alerts  │
           └────────────────────┘
```

---

## 📂 Project Structure

```
visp/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── stream.py          # WebSocket video stream endpoint
│   │   │   ├── events.py          # Event log CRUD
│   │   │   └── health.py          # Health & readiness probes
│   │   └── dependencies.py        # FastAPI DI helpers
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   ├── logging.py             # Structured logging
│   │   └── security.py            # API key auth
│   ├── models/
│   │   ├── base.py                # Abstract detector interface
│   │   ├── mvit.py                # MViT-v2 wrapper
│   │   ├── vivit.py               # ViViT-B wrapper
│   │   └── r2plus1d.py            # R(2+1)D wrapper
│   ├── services/
│   │   ├── inference.py           # Frame batching + model dispatch
│   │   ├── alert.py               # Notification dispatch
│   │   ├── event_queue.py         # Redis event publisher
│   │   └── ppe_detector.py        # PPE-specific detection logic
│   └── utils/
│       ├── frame_processor.py     # Pre/post-processing
│       └── zone_manager.py        # Polygon zone logic
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoFeed.jsx      # Live WebSocket video viewer
│   │   │   ├── EventLog.jsx       # Scrollable event timeline
│   │   │   ├── AlertBadge.jsx     # Severity badge component
│   │   │   ├── HeatmapOverlay.jsx # Zone heatmap
│   │   │   └── StatCard.jsx       # KPI stat card
│   │   ├── hooks/
│   │   │   ├── useStream.js       # WebSocket stream hook
│   │   │   └── useEvents.js       # Event polling hook
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Main monitoring view
│   │   │   ├── Events.jsx         # Event history & filters
│   │   │   └── Settings.jsx       # Config & zone editor
│   │   └── services/
│   │       └── api.js             # Axios API client
│   └── public/
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── scripts/
│   ├── export_onnx.py             # Export PyTorch → ONNX
│   ├── benchmark.py               # Latency / throughput bench
│   └── seed_demo.py               # Seed Redis with demo events
├── tests/
│   ├── unit/
│   │   ├── test_inference.py
│   │   ├── test_zone_manager.py
│   │   └── test_alert_service.py
│   └── integration/
│       └── test_websocket_stream.py
├── docs/
│   ├── architecture.md
│   ├── deployment.md
│   └── api_reference.md
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── docker-publish.yml
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- (Optional) NVIDIA GPU with CUDA 11.8+

### 1. Clone & configure

```bash
git clone https://github.com/your-org/visp.git
cd visp
cp .env.example .env
# Edit .env with your settings
```

### 2. Run with Docker (recommended)

```bash
docker compose up --build
```

The stack will be available at:
- **Dashboard** → http://localhost:3000
- **API** → http://localhost:8000
- **API Docs** → http://localhost:8000/docs

### 3. Run locally (development)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install && npm run dev
```

---

## 📡 API Reference

### WebSocket — Live Stream

```
WS /ws/stream/{camera_id}
```

Connect and send raw video frames (JPEG bytes). Receive JSON events:

```json
{
  "timestamp": "2025-04-29T14:32:11Z",
  "camera_id": "cam-01",
  "event_type": "violence_detected",
  "confidence": 0.91,
  "bounding_box": [120, 80, 320, 400],
  "zone": "warehouse-floor",
  "frame_id": 4821
}
```

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/events` | List events (filterable) |
| `GET` | `/api/events/{id}` | Single event detail |
| `GET` | `/api/cameras` | Registered camera list |
| `POST` | `/api/cameras` | Register a new stream |
| `POST` | `/api/upload` | Analyze a video file |
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/ready` | Readiness probe |

Full OpenAPI spec at `/docs` when running.

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Unit only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

---

## 📦 ONNX Export (Edge Deployment)

Export any model to ONNX for deployment on CPU, Jetson Nano, or Raspberry Pi:

```bash
python scripts/export_onnx.py \
  --model mvit \
  --checkpoint checkpoints/mvit_violence_v2.pt \
  --output models/mvit_violence.onnx \
  --optimize  # applies graph optimizations
```

Benchmarking:

```bash
python scripts/benchmark.py --model models/mvit_violence.onnx --backend onnx
```

---

## ⚙️ Configuration

All settings are driven by environment variables (`.env`):

```env
# Model
MODEL_BACKEND=mvit           # mvit | vivit | r2plus1d | onnx
ONNX_MODEL_PATH=models/mvit_violence.onnx
CONFIDENCE_THRESHOLD=0.75
CLIP_LENGTH=16               # frames per inference window

# Streaming
MAX_CONCURRENT_STREAMS=8
FRAME_SKIP=2                 # process every Nth frame

# Redis
REDIS_URL=redis://localhost:6379/0
EVENT_TTL_SECONDS=86400

# Alerts
ALERT_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_EMAIL_TO=safety@yourcompany.com
SMTP_HOST=smtp.yourprovider.com

# Auth
API_KEY=your-secret-api-key
```

---

## 🚢 Deployment

See [`docs/deployment.md`](docs/deployment.md) for full guides on:
- **Docker Compose** (single server)
- **Kubernetes** (Helm chart included)
- **AWS ECS / GCP Cloud Run**
- **Jetson Nano** (ONNX edge deployment)

---

## 🛣️ Roadmap

- [x] MViT / R(2+1)D inference pipeline
- [x] WebSocket real-time stream
- [x] React dashboard MVP
- [x] ONNX export & optimization
- [ ] Multi-camera correlation engine
- [ ] Anomaly detection (unsupervised)
- [ ] Mobile app (React Native)
- [ ] On-prem Helm chart (production-hardened)
- [ ] Fine-tuning UI (label studio integration)

---

<div align="center">
Built with PyTorch Video · FastAPI · React · ONNX Runtime
</div>
