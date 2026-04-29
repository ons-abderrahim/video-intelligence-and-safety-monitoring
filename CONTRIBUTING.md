# Contributing to VISP

Thanks for your interest! We welcome bug reports, feature ideas, and pull requests.

## Setup

```bash
git clone https://github.com/your-org/visp.git
cd visp

# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

## Branching

- `main` — stable releases
- `develop` — active development
- `feat/*`, `fix/*`, `chore/*` — feature / fix / housekeeping branches

## Running Tests

```bash
pytest tests/ -v            # all tests
pytest tests/unit/ -v       # unit only
```

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] New code has unit tests
- [ ] `ruff check` passes
- [ ] PR description explains *why*, not just *what*

## Reporting Issues

Please include:
1. VISP version / commit SHA
2. Steps to reproduce
3. Expected vs. actual behaviour
4. Logs (with sensitive data redacted)
