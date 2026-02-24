# Installation

## Requirements

- Python 3.11+
- pip

## Option A: Minimal Runtime

```bash
pip install -e .
```

## Option B: Standard Development Setup

```bash
pip install -e .[dev]
```

## Option C: Full Setup (DB + visualization extras)

```bash
pip install -e .[dev,db,viz]
```

## Reproducible Runtime Requirements

For deployment targets that require a pinned list (e.g., Streamlit Cloud):

```bash
pip install -r requirements.txt
```

## Verify Installation

```bash
pytest tests/test_physics.py -v
ruff check frap2025/ tests/
```

## Launch UI

```bash
streamlit run app.py --server.port 5000
```
