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

## Option D: Advanced Biophysics / ML Stack

Includes optional dependencies for:

- Physics-Informed Neural Networks (PyTorch)
- Deep optical flow RAFT backends (TorchVision)
- Optimal transport (POT)
- Non-parametric Bayesian SPT inference (Pyro)

```bash
pip install -e .[advanced]
```

For full development with all extras:

```bash
pip install -e .[dev,db,viz,advanced]
```

## Reproducible Runtime Requirements

For deployment targets that require a pinned list (e.g., Streamlit Cloud):

```bash
pip install -r requirements.txt
```

`requirements.txt` is a pinned runtime snapshot; the canonical editable/development dependency definitions live in `pyproject.toml`.

## Verify Installation

```bash
pytest tests/test_physics.py -v
ruff check frap2025/ tests/
```

## Launch UI

```bash
streamlit run app.py --server.port 5000
```
