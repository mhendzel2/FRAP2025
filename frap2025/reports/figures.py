"""Figure export helpers."""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, filename_stem: str, formats: list[str] | tuple[str, ...] = ("svg", "png"), dpi: int = 300) -> list[str]:
    """Save one figure into multiple publication formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    filename_stem : str
        Output path without extension.
    formats : sequence of str
        Format list, e.g. `("svg", "png")`.
    dpi : int
        Raster DPI for non-vector outputs.
    """
    if not formats:
        raise ValueError("formats must include at least one output format.")

    stem_path = Path(filename_stem)
    saved_paths: list[str] = []
    for fmt in formats:
        ext = str(fmt).lower().lstrip(".")
        out_path = stem_path.with_suffix(f".{ext}")
        if ext == "svg":
            fig.savefig(out_path, format="svg", bbox_inches="tight")
        else:
            fig.savefig(out_path, format=ext, dpi=dpi, bbox_inches="tight")
        saved_paths.append(str(out_path))
    return saved_paths


__all__ = ["save_figure"]
