"""Canonical package entrypoint for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    """Execute the preserved legacy Streamlit application script."""
    ui_dir = Path(__file__).resolve().parent
    runpy.run_path(str(ui_dir / "legacy_app.py"), run_name="__main__")
