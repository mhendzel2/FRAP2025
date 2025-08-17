import io
import logging
from typing import Optional, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)

try:
    import roifile  # optional
except ImportError:  # graceful if not installed
    roifile = None


def import_imagej_roi(roi_data: bytes) -> Optional[Dict[str, Any]]:
    """Parse an ImageJ .roi file and return metadata dictionary.
    Returns None if parsing fails or roifile unavailable.
    """
    if roifile is None:
        st.error("roifile package not installed. Install 'roifile' to import ImageJ ROI files.")
        return None
    try:
        roi = roifile.roiread(io.BytesIO(roi_data))
        return {
            'name': getattr(roi, 'name', ''),
            'type': getattr(roi, 'roitype', None).name if getattr(roi, 'roitype', None) else 'Unknown',
            'left': roi.left,
            'top': roi.top,
            'right': roi.right,
            'bottom': roi.bottom,
            'width': roi.width,
            'height': roi.height,
            'coordinates': roi.coordinates().tolist() if hasattr(roi, 'coordinates') else []
        }
    except Exception as e:
        logger.error(f"ROI import failed: {e}")
        st.error(f"Failed to import ROI file: {e}")
        return None
