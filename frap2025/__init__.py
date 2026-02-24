"""FRAP2025 package."""

from __future__ import annotations

from pathlib import Path
import sys

_LEGACY_MODULE_DIR = Path(__file__).resolve().parents[1] / "legacy" / "modules"
if _LEGACY_MODULE_DIR.exists():
    legacy_path = str(_LEGACY_MODULE_DIR)
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)

from . import calibration, core, database, frap_fitting, image, io, reports, spt_models, stats, ui  # noqa: E402
from . import material_mechanics, optical_flow_analysis, population_analysis  # noqa: E402

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "core",
    "io",
    "image",
    "stats",
    "reports",
    "database",
    "calibration",
    "ui",
    "frap_fitting",
    "spt_models",
    "population_analysis",
    "material_mechanics",
    "optical_flow_analysis",
]
