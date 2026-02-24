# Contributing

## Local Setup

- Python 3.11+
- Install editable with dev extras:
  - `pip install -e .[dev]`

## Required Checks

- `ruff check frap2025/ tests/`
- `pytest tests/test_physics.py -v`
- `pytest tests/ -v`

## Code Standards

- Use explicit units in variable names (`tau_s`, `D_um2_per_s`, `w_um`).
- No silent fit failures; emit warnings with quality flags.
- Keep normalization mode explicit and validated.

## Data Policy

- Commit only small synthetic/public samples under `sample_data/`.
- Keep experimental data local under `data/`.
