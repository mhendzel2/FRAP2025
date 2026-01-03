from __future__ import annotations

import inspect
from typing import Any, Callable


def _supports_kwarg(fn: Callable[..., Any], kwarg: str) -> bool:
    try:
        return kwarg in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _wrap_width_to_container(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap Streamlit functions to accept `width=` on older Streamlit.

    Streamlit has deprecated `use_container_width` in favor of `width`:
      - width='stretch'  <=> use_container_width=True
      - width='content'  <=> use_container_width=False

    On newer Streamlit versions that already support `width`, return the original.
    """

    if _supports_kwarg(fn, "width"):
        return fn

    def wrapped(*args: Any, width: str | int | None = None, **kwargs: Any):
        if width is not None and "use_container_width" not in kwargs:
            # Only translate the string modes we use.
            if width == "stretch":
                kwargs["use_container_width"] = True
            elif width == "content":
                kwargs["use_container_width"] = False
        return fn(*args, **kwargs)

    return wrapped


def patch_streamlit_width(st_module: Any) -> None:
    """Patch common Streamlit APIs so `width=` works across versions."""

    for attr in (
        "button",
        "download_button",
        "dataframe",
        "plotly_chart",
        "line_chart",
        "bar_chart",
        "area_chart",
        "pyplot",
    ):
        if hasattr(st_module, attr):
            fn = getattr(st_module, attr)
            if callable(fn):
                setattr(st_module, attr, _wrap_width_to_container(fn))
