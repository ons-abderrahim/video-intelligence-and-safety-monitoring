# Deployment Guide

## Docker Compose (single server)

The fastest path to production.

```bash
git clone https://github.com/your-org/visp.git && cd visp
cp .env.example .env   # fill in API_KEY and alert settings
docker compose up -d --build
```

Services exposed:
| Service | Port |
|---|---|
| Dashboard | 3000 |
| API + WebSocket | 8000 |
| Prometheus | 9090 |
| Grafana | 3001 |

## Kubernetes (Helm)

```bash
helm repo add visp https://your-org.github.io/visp-helm
helm install visp visp/visp \
  --set backend.apiKey=<secret> \
  --set backend.modelBackend=onnx \
  --set redis.enabled=true \
  --set postgres.enabled=true
```

## Edge Deployment (Jetson Nano / Raspberry Pi)

1. Export model to ONNX:
```bash
python scripts/export_onnx.py --model mvit --optimize
```

2. Set in `.env`:
```
MODEL_BACKEND=onnx
ONNX_MODEL_PATH=models/visp_model.onnx
DEVICE=cpu
MAX_CONCURRENT_STREAMS=2
```

3. Run backend only (no GPU needed):
```bash
docker compose up backend redis
```

## Environment Variables

All configuration is documented in `.env.example`.
Key production settings:

- `API_KEY` — required, keep secret
- `CONFIDENCE_THRESHOLD` — tune to reduce false positives
- `MAX_CONCURRENT_STREAMS` — set based on server capacity
- `DEVICE=cuda` — enable GPU inference if available
