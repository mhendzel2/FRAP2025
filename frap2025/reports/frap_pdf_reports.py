"""Compatibility proxy for legacy `frap_pdf_reports.py`."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LEGACY = import_module("frap_pdf_reports")
_PUBLIC = [name for name in dir(_LEGACY) if not name.startswith("_")]


def __getattr__(name: str) -> Any:
    if name in _PUBLIC:
        return getattr(_LEGACY, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC))


__all__ = _PUBLIC
