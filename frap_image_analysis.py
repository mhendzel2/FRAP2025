"""
FRAP Image Analysis Module
Enhanced image processing capabilities for direct analysis of raw microscopy data
"""

import numpy as np
import pandas as pd
import cv2
import tifffile
from skimage import filters, measure, morphology, segmentation
from skimage.restoration import rolling_ball
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple, List, Dict, Optional
import os
from pathlib import Path
from frap_utils import import_imagej_roi
from frap_core import FRAPAnalysisCore

class FRAPImageAnalyzer:
    """
    Comprehensive image analysis for FRAP experiments including:
    - Multi-format image loading (TIFF, multi-page TIFF, image sequences)
    - Automated bleach spot detection and tracking
    - ROI selection and intensity extraction
    - Background correction and normalization
    - PSF calibration and spatial analysis
    """
    
    def __init__(self):
        """Initialize state containers for image analysis."""
        self.image_stack = None
        self.metadata = {}
        self.rois = {}
        self.bleach_frame = None
        self.bleach_coordinates = None
        self.time_points = None
        self.pixel_size = 1.0  # micrometers per pixel
        self.time_interval = 1.0  # seconds per frame
        # Dynamic tracking (list of (x,y) centers per frame when enabled)
        self.tracked_centers = None
        self.stabilization_results = None
        
    def load_image_stack(self, file_path: str) -> bool:
        """
        Load image stack from various formats
        
        Parameters:
        -----------
        file_path : str
            Path to image file or directory
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if file_path.is_file():
                # Single file - check extension
                if file_path.suffix.lower() in ['.tif', '.tiff']:
                    self.image_stack = tifffile.imread(str(file_path))
                    self._extract_tiff_metadata(str(file_path))
                elif file_path.suffix.lower() in ['.czi']:
                    st.warning("CZI format detected. Install czifile package for optimal support.")
                    return False
                elif file_path.suffix.lower() in ['.lif']:
                    st.warning("LIF format detected. Install readlif package for optimal support.")
                    return False
                else:
                    # Try as single image
                    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.image_stack = np.expand_dims(img, axis=0)
                    else:
                        return False
            
            elif file_path.is_dir():
                # Directory of images - load sequence
                image_files = sorted([f for f in file_path.glob('*') 
                                    if f.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']])
                if not image_files:
                    return False
                
                # Load first image to get dimensions
                first_img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
                if first_img is None:
                    return False
                
                # Initialize stack
                self.image_stack = np.zeros((len(image_files), *first_img.shape), dtype=first_img.dtype)
                self.image_stack[0] = first_img
                
                # Load remaining images
                for i, img_file in enumerate(image_files[1:], 1):
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.shape == first_img.shape:
                        self.image_stack[i] = img
            
            # Ensure 3D stack (time, height, width)
            if self.image_stack.ndim == 2:
                self.image_stack = np.expand_dims(self.image_stack, axis=0)
            
            # Generate time points if not available
            if self.time_points is None:
                self.time_points = np.arange(len(self.image_stack)) * self.time_interval
            
            return True
            
        except Exception as e:
            st.error(f"Error loading image stack: {str(e)}")
            return False
    
    def _extract_tiff_metadata(self, file_path: str):
        """Extract metadata from TIFF files"""
        try:
            with tifffile.TiffFile(file_path) as tif:
                # Extract basic metadata
                self.metadata = {}
                
                # Try to get pixel size and time information
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                    if 'spacing' in metadata:
                        self.time_interval = metadata.get('spacing', 1.0)
                    if 'unit' in metadata:
                        if metadata['unit'] == 'micron':
                            # Extract pixel size if available
                            pass
                
                # Store basic info
                self.metadata['n_frames'] = len(tif.pages)
                self.metadata['shape'] = tif.pages[0].shape
                self.metadata['dtype'] = tif.pages[0].dtype
                
        except Exception as e:
            st.warning(f"Could not extract TIFF metadata: {str(e)}")
    
    def detect_bleach_event(self, roi_coords: Optional[Tuple] = None, 
                           method: str = 'intensity_drop') -> Tuple[int, Tuple[int, int]]:
        """
        Automatically detect bleach event timing and location
        
        Parameters:
        -----------
        roi_coords : tuple, optional
            (x, y, width, height) for region to analyze
        method : str
            Detection method: 'intensity_drop', 'gradient', or 'manual'
            
        Returns:
        --------
        tuple
            (bleach_frame, (x, y)) coordinates of bleach center
        """
        if self.image_stack is None:
            raise ValueError("No image stack loaded")
        
        if method == 'intensity_drop':
            # Calculate mean intensity over time
            if roi_coords:
                x, y, w, h = roi_coords
                roi_stack = self.image_stack[:, y:y+h, x:x+w]
                mean_intensities = np.mean(roi_stack, axis=(1, 2))
            else:
                mean_intensities = np.mean(self.image_stack, axis=(1, 2))
            
            # Find largest intensity drop
            intensity_diffs = np.diff(mean_intensities)
            bleach_frame = np.argmin(intensity_diffs) + 1
            
            # Find bleach spot location in that frame
            if bleach_frame < len(self.image_stack):
                bleach_image = self.image_stack[bleach_frame]
                
                # Find darkest region (bleach spot)
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(bleach_image, (5, 5), 1)
                
                # Find minimum intensity location
                min_loc = np.unravel_index(np.argmin(blurred), blurred.shape)
                bleach_coords = (min_loc[1], min_loc[0])  # (x, y)
        
        elif method == 'gradient':
            # Detect based on temporal gradient
            temporal_gradient = np.gradient(self.image_stack.astype(float), axis=0)
            
            # Find frame with largest negative gradient
            gradient_magnitude = np.mean(np.abs(temporal_gradient), axis=(1, 2))
            bleach_frame = np.argmax(gradient_magnitude)
            
            # Find location of maximum gradient change
            gradient_frame = temporal_gradient[bleach_frame]
            max_grad_loc = np.unravel_index(np.argmax(np.abs(gradient_frame)), gradient_frame.shape)
            bleach_coords = (max_grad_loc[1], max_grad_loc[0])
        
        else:
            # Manual detection - return None to prompt user selection
            return None, None
        
        self.bleach_frame = bleach_frame
        self.bleach_coordinates = bleach_coords
        
        return bleach_frame, bleach_coords
    
    def define_rois(self, bleach_center: Tuple[int, int], 
                   bleach_radius: int = 10,
                   background_distance: int = 50) -> Dict[str, Dict]:
        """
        Automatically define ROIs based on bleach spot location
        
        Parameters:
        -----------
        bleach_center : tuple
            (x, y) coordinates of bleach center
        bleach_radius : int
            Radius of bleach ROI in pixels
        background_distance : int
            Distance from bleach center for background ROI
            
        Returns:
        --------
        dict
            Dictionary of ROI definitions
        """
        if self.image_stack is None:
            raise ValueError("No image stack loaded")
        
        height, width = self.image_stack.shape[1:]
        x_center, y_center = bleach_center
        
        # ROI 1: Bleach spot (circular)
        roi1_mask = np.zeros((height, width), dtype=bool)
        y_grid, x_grid = np.ogrid[:height, :width]
        mask = (x_grid - x_center)**2 + (y_grid - y_center)**2 <= bleach_radius**2
        roi1_mask[mask] = True
        
        # ROI 2: Reference region (annulus around bleach spot)
        ref_inner_radius = bleach_radius + 5
        ref_outer_radius = bleach_radius + 20
        roi2_mask = np.zeros((height, width), dtype=bool)
        annulus_mask = ((x_grid - x_center)**2 + (y_grid - y_center)**2 >= ref_inner_radius**2) & \
                      ((x_grid - x_center)**2 + (y_grid - y_center)**2 <= ref_outer_radius**2)
        roi2_mask[annulus_mask] = True
        
        # ROI 3: Background region (distant from bleach spot)
        bg_x = max(background_distance, min(width - background_distance, x_center + background_distance))
        bg_y = max(background_distance, min(height - background_distance, y_center + background_distance))
        bg_radius = 15
        
        roi3_mask = np.zeros((height, width), dtype=bool)
        bg_mask = (x_grid - bg_x)**2 + (y_grid - bg_y)**2 <= bg_radius**2
        roi3_mask[bg_mask] = True
        
        self.rois = {
            'ROI1': {
                'mask': roi1_mask,
                'center': (x_center, y_center),
                'radius': bleach_radius,
                'type': 'bleach_spot'
            },
            'ROI2': {
                'mask': roi2_mask,
                'center': (x_center, y_center),
                'inner_radius': ref_inner_radius,
                'outer_radius': ref_outer_radius,
                'type': 'reference'
            },
            'ROI3': {
                'mask': roi3_mask,
                'center': (bg_x, bg_y),
                'radius': bg_radius,
                'type': 'background'
            }
        }
        
        return self.rois

    def add_roi_from_import(self, roi_info: Dict, roi_type: str) -> bool:
        """
        Add an ROI from an imported ImageJ ROI file.

        Parameters:
        -----------
        roi_info : dict
            Dictionary of ROI properties from import_imagej_roi.
        roi_type : str
            The type of ROI ('bleach_spot', 'reference', 'background').

        Returns:
        --------
        bool
            True if successful, False otherwise.
        """
        if self.image_stack is None:
            st.error("Cannot add ROI without a loaded image.")
            return False

        height, width = self.image_stack.shape[1:]
        mask = np.zeros((height, width), dtype=bool)

        # Create a mask from the coordinates
        # The coordinates are [[y1, x1], [y2, x2], ...]
        coords = np.array(roi_info['coordinates']).astype(int)

        # For polygon, fill the polygon to create a mask
        if roi_info['type'].lower() in ['polygon', 'freehand']:
            cv2.fillPoly(mask.astype(np.uint8), [coords[:, ::-1]], 1) # cv2 expects (x,y)
            mask = mask.astype(bool)
        # For rectangle, create a rectangle mask
        elif roi_info['type'].lower() == 'rectangle':
            left, top, w, h = roi_info['left'], roi_info['top'], roi_info['width'], roi_info['height']
            mask[top:top+h, left:left+w] = True
        # For oval, create an ellipse mask
        elif roi_info['type'].lower() == 'oval':
            # Get center and axes from bounding box
            cx = roi_info['left'] + roi_info['width'] // 2
            cy = roi_info['top'] + roi_info['height'] // 2
            ax1 = roi_info['width'] // 2
            ax2 = roi_info['height'] // 2
            y_grid, x_grid = np.ogrid[:height, :width]
            ellipse_mask = ((x_grid - cx)**2 / ax1**2 + (y_grid - cy)**2 / ax2**2 <= 1)
            mask[ellipse_mask] = True
        else:
            st.warning(f"Unsupported ROI type '{roi_info['type']}' for mask creation. Using bounding box.")
            left, top, w, h = roi_info['left'], roi_info['top'], roi_info['width'], roi_info['height']
            mask[top:top+h, left:left+w] = True

        roi_name = f"ROI{len(self.rois) + 1}"
        self.rois[roi_name] = {
            'mask': mask,
            'center': (roi_info['left'] + roi_info['width'] // 2, roi_info['top'] + roi_info['height'] // 2),
            'type': roi_type,
            'name': roi_info['name']
        }
        st.success(f"Added '{roi_info['name']}' as {roi_name} ({roi_type})")
        return True
    
    def extract_intensity_profiles(self, apply_background_correction: bool = True, enable_stabilization: bool = True) -> pd.DataFrame:
        """
        Extract intensity profiles from defined ROIs, with optional motion stabilization.
        
        Parameters:
        -----------
        apply_background_correction : bool
            Whether to apply background correction.
        enable_stabilization : bool
            Whether to perform motion stabilization before intensity extraction.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time points, intensity values, and stabilization metrics.
        """
        if self.image_stack is None or not self.rois:
            raise ValueError("No image stack or ROIs defined")

        n_frames, height, width = self.image_stack.shape
        
        roi_map = {}
        for name, data in self.rois.items():
            if data['type'] == 'bleach_spot' and 'ROI1' not in roi_map:
                roi_map['ROI1'] = name
            elif data['type'] == 'reference' and 'ROI2' not in roi_map:
                roi_map['ROI2'] = name
            elif data['type'] == 'background' and 'ROI3' not in roi_map:
                roi_map['ROI3'] = name
        
        if len(roi_map) != 3:
            st.error("Please define exactly one of each ROI type: 'bleach_spot', 'reference', and 'background'.")
            return pd.DataFrame()

        # --- Motion Stabilization ---
        motion_qc_flag = False
        motion_qc_reason = "Stabilization not enabled."
        stabilized_stack = self.image_stack

        if enable_stabilization:
            try:
                st.info("Performing motion stabilization...")
                bleach_roi_data = self.rois[roi_map['ROI1']]

                self.stabilization_results = FRAPAnalysisCore.motion_compensate_stack(
                    stack=self.image_stack,
                    init_center=self.bleach_coordinates,
                    radius=bleach_roi_data['radius'],
                    pixel_size_um=self.pixel_size,
                    use_optical_flow=True,
                    do_global=True,
                    kalman=True
                )

                warnings = self.stabilization_results.get('warnings', [])
                if not warnings:
                    motion_qc_flag = True
                    motion_qc_reason = "Stabilization successful."
                    stabilized_stack = self.stabilization_results['stabilized_stack']
                    # Use the tracked centers from the more advanced stabilization
                    self.tracked_centers = [d['centroid'] for d in self.stabilization_results['roi_trace']]
                    st.success("Motion stabilization successful.")
                else:
                    motion_qc_flag = False
                    motion_qc_reason = "; ".join(warnings)
                    st.warning(f"Stabilization finished with warnings: {motion_qc_reason}. Falling back to static ROI.")

            except Exception as e:
                motion_qc_flag = False
                motion_qc_reason = f"Stabilization failed: {str(e)}"
                st.error(f"Motion stabilization failed: {e}. Falling back to static ROI.")
                self.stabilization_results = None

        # --- Intensity Extraction ---
        intensities = {'ROI1': np.zeros(n_frames), 'ROI2': np.zeros(n_frames), 'ROI3': np.zeros(n_frames)}
        
        ref_mask = self.rois[roi_map['ROI2']]['mask']
        bg_mask = self.rois[roi_map['ROI3']]['mask']
        y_grid, x_grid = np.ogrid[:height, :width]

        for t in range(n_frames):
            frame = stabilized_stack[t].astype(float)
            if apply_background_correction:
                frame = self._apply_background_correction(frame)

            # If stabilization was successful and tracked centers are available, use dynamic ROI
            if motion_qc_flag and self.tracked_centers:
                center_x, center_y = self.tracked_centers[t]['x'], self.tracked_centers[t]['y']
                radius = self.rois[roi_map['ROI1']]['radius']
                bleach_mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
            else: # Fallback to static ROI
                bleach_mask = self.rois[roi_map['ROI1']]['mask']

            intensities['ROI1'][t] = np.mean(frame[bleach_mask]) if np.any(bleach_mask) else 0
            intensities['ROI2'][t] = np.mean(frame[ref_mask]) if np.any(ref_mask) else 0
            intensities['ROI3'][t] = np.mean(frame[bg_mask]) if np.any(bg_mask) else 0

        # --- Assemble DataFrame ---
        df = pd.DataFrame({
            'time': self.time_points[:n_frames],
            'ROI1': intensities['ROI1'],
            'ROI2': intensities['ROI2'],
            'ROI3': intensities['ROI3']
        })

        # Add stabilization metrics to the dataframe
        if self.stabilization_results:
            trace = self.stabilization_results['roi_trace']
            df['roi_centroid_x'] = [d['centroid']['x'] for d in trace]
            df['roi_centroid_y'] = [d['centroid']['y'] for d in trace]
            df['roi_radius_per_frame'] = [d['applied_radius'] for d in trace]
            df['total_drift_um'] = self.stabilization_results.get('drift_um', np.nan)

            displacements_px = [d['displacement_px'] for d in trace]
            mean_shift_um = np.mean(displacements_px) * self.pixel_size if self.pixel_size else np.nan
            df['mean_framewise_shift_um'] = mean_shift_um
        else:
            # Fill with NaNs if stabilization was not run or failed
            df['roi_centroid_x'] = np.nan
            df['roi_centroid_y'] = np.nan
            df['roi_radius_per_frame'] = np.nan
            df['total_drift_um'] = np.nan
            df['mean_framewise_shift_um'] = np.nan

        df['motion_qc_flag'] = motion_qc_flag
        df['motion_qc_reason'] = motion_qc_reason
        
        return df

    def track_bleach_spot(self, search_window: int = 20, invert: bool = True) -> List[Tuple[int, int]]:
        """Dynamically track bleach spot center across frames.

        Uses optional inversion so that dark bleach spot becomes bright, then
        computes an intensity-weighted centroid within a local window around
        the previous center.

        Parameters
        ----------
        search_window : int
            Half-size of square window (pixels) used for local search.
        invert : bool
            If True, invert each frame (max - frame) prior to centroid calc.

        Returns
        -------
        list[tuple]
            List of (x, y) centers per frame.
        """
        if self.image_stack is None or self.bleach_coordinates is None:
            raise ValueError("Image stack and initial bleach coordinates required before tracking")

        n_frames, height, width = self.image_stack.shape
        centers: List[Tuple[int, int]] = []
        cx, cy = self.bleach_coordinates  # initial center (x,y)
        centers.append((cx, cy))

        for t in range(1, n_frames):
            frame = self.image_stack[t].astype(float)
            if invert:
                frame = frame.max() - frame
            x_min = max(0, cx - search_window)
            x_max = min(width, cx + search_window + 1)
            y_min = max(0, cy - search_window)
            y_max = min(height, cy + search_window + 1)
            sub = frame[y_min:y_max, x_min:x_max]
            if sub.size == 0:
                centers.append((cx, cy))
                continue
            # Intensity-weighted centroid
            y_ind, x_ind = np.indices(sub.shape)
            weights = sub - sub.min()
            if weights.sum() <= 0:
                new_cx, new_cy = cx, cy
            else:
                wx = (weights * x_ind).sum() / weights.sum()
                wy = (weights * y_ind).sum() / weights.sum()
                new_cx = int(round(x_min + wx))
                new_cy = int(round(y_min + wy))
            # Constrain to image bounds
            new_cx = max(0, min(width - 1, new_cx))
            new_cy = max(0, min(height - 1, new_cy))
            centers.append((new_cx, new_cy))
            cx, cy = new_cx, new_cy

        self.tracked_centers = centers
        return centers
    
    def _apply_background_correction(self, image: np.ndarray, 
                                   method: str = 'rolling_ball') -> np.ndarray:
        """
        Apply background correction to image
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        method : str
            Background correction method
            
        Returns:
        --------
        np.ndarray
            Background-corrected image
        """
        if method == 'rolling_ball':
            # Rolling ball background subtraction
            try:
                background = rolling_ball(image, radius=50)
                corrected = image - background
                return np.maximum(corrected, 0)  # Ensure non-negative values
            except:
                # Fallback to simple background subtraction
                background = filters.gaussian(image, sigma=20)
                return np.maximum(image - background, 0)
        
        elif method == 'gaussian_blur':
            # Simple Gaussian background estimation
            background = filters.gaussian(image, sigma=20)
            return np.maximum(image - background, 0)
        
        else:
            return image
    
    def estimate_psf_parameters(self, bleach_frame_offset: int = 1) -> Dict[str, float]:
        """
        Estimate Point Spread Function parameters from bleach spot
        
        Parameters:
        -----------
        bleach_frame_offset : int
            Frames after bleach to analyze
            
        Returns:
        --------
        dict
            PSF parameters including effective radius
        """
        if self.image_stack is None or self.bleach_frame is None:
            raise ValueError("No image stack or bleach frame identified")
        
        # Get image shortly after bleach
        psf_frame_idx = min(self.bleach_frame + bleach_frame_offset, len(self.image_stack) - 1)
        psf_image = self.image_stack[psf_frame_idx].astype(float)
        
        # Extract region around bleach spot
        x_center, y_center = self.bleach_coordinates
        crop_size = 40
        x_start = max(0, x_center - crop_size//2)
        x_end = min(psf_image.shape[1], x_center + crop_size//2)
        y_start = max(0, y_center - crop_size//2)
        y_end = min(psf_image.shape[0], y_center + crop_size//2)
        
        cropped = psf_image[y_start:y_end, x_start:x_end]
        
        # Find local minimum (bleach spot center in cropped image)
        local_center = np.unravel_index(np.argmin(cropped), cropped.shape)
        
        # Calculate radial profile
        y_indices, x_indices = np.ogrid[:cropped.shape[0], :cropped.shape[1]]
        distances = np.sqrt((x_indices - local_center[1])**2 + (y_indices - local_center[0])**2)
        
        # Bin by distance and calculate mean intensity
        max_distance = min(crop_size//2 - 2, 15)
        distance_bins = np.arange(0, max_distance, 0.5)
        radial_profile = []
        
        for i in range(len(distance_bins) - 1):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
            if np.any(mask):
                radial_profile.append(np.mean(cropped[mask]))
            else:
                radial_profile.append(np.nan)
        
        radial_profile = np.array(radial_profile)
        radial_distances = (distance_bins[:-1] + distance_bins[1:]) / 2
        
        # Estimate effective bleach radius (where intensity recovers to ~80% of surrounding)
        valid_mask = ~np.isnan(radial_profile)
        if np.sum(valid_mask) > 3:
            # Find radius where intensity reaches 80% of far-field value
            far_field_intensity = np.nanmean(radial_profile[-3:])  # Average of last 3 points
            min_intensity = np.nanmin(radial_profile)
            threshold_intensity = min_intensity + 0.8 * (far_field_intensity - min_intensity)
            
            # Find first radius where intensity exceeds threshold
            threshold_indices = np.where(radial_profile > threshold_intensity)[0]
            if len(threshold_indices) > 0:
                effective_radius_pixels = radial_distances[threshold_indices[0]]
            else:
                effective_radius_pixels = 3.0  # Default fallback
        else:
            effective_radius_pixels = 3.0
        
        # Convert to micrometers
        effective_radius_um = effective_radius_pixels * self.pixel_size
        
        return {
            'effective_radius_pixels': effective_radius_pixels,
            'effective_radius_um': effective_radius_um,
            'radial_profile': radial_profile,
            'radial_distances': radial_distances,
            'pixel_size_um': self.pixel_size
        }
    
    def create_analysis_summary(self) -> Dict:
        """
        Create comprehensive analysis summary
        
        Returns:
        --------
        dict
            Summary of image analysis results
        """
        summary = {
            'image_info': {
                'n_frames': len(self.image_stack) if self.image_stack is not None else 0,
                'image_shape': self.image_stack.shape[1:] if self.image_stack is not None else None,
                'pixel_size_um': self.pixel_size,
                'time_interval_s': self.time_interval
            },
            'bleach_detection': {
                'bleach_frame': self.bleach_frame,
                'bleach_coordinates': self.bleach_coordinates
            },
            'roi_info': {
                'n_rois': len(self.rois),
                'roi_types': [roi['type'] for roi in self.rois.values()] if self.rois else []
            }
        }
        
        return summary

    # (Conflict resolution note) Removed legacy optional import of frap_roi_utils.
    # The function import_imagej_roi is now provided by frap_utils; no duplicate import needed.
import tempfile
def create_image_analysis_interface(dm):
    """Create Streamlit interface for image analysis"""
    st.header("🔬 FRAP Image Analysis")
    st.write("Direct analysis of raw microscopy images with automated or imported ROI detection")

    if 'frap_analyzer' not in st.session_state:
        st.session_state.frap_analyzer = FRAPImageAnalyzer()
    analyzer = st.session_state.frap_analyzer
    if 'image_analysis_df' not in st.session_state:
        st.session_state.image_analysis_df = None

    uploaded_file = st.file_uploader(
        "Upload FRAP Image Stack",
        type=['tif', 'tiff'],
        help="Upload TIFF files or image sequences"
    )

    if uploaded_file:
        if not hasattr(analyzer, 'file_name') or analyzer.file_name != uploaded_file.name:
            with st.spinner("Loading image..."):
                st.session_state.image_analysis_df = None # Reset previous analysis
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if analyzer.load_image_stack(temp_path):
                    analyzer.file_name = uploaded_file.name
                    st.success(f"✅ Loaded image stack: {analyzer.image_stack.shape}")
                    os.remove(temp_path)
                else:
                    st.error("Failed to load image stack.")
                    st.stop()

    if analyzer.image_stack is not None:
        st.subheader("Analysis Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            pixel_size = st.number_input("Pixel Size (μm)", value=0.1, min_value=0.01, max_value=1.0, step=0.01, key="ia_pixel_size")
            time_interval = st.number_input("Time Interval (s)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, key="ia_time_interval")
        with col2:
            bleach_radius = st.number_input("Bleach Spot Radius (pixels)", value=10, min_value=3, max_value=50, key="ia_bleach_radius")
            detection_method = st.selectbox("Bleach Detection", ["intensity_drop", "gradient"], key="ia_detection_method")
        with col3:
            enable_tracking = st.checkbox("Enable ROI Tracking", value=True, help="Track bleach spot over time with image inversion.", key="ia_enable_tracking")

        analyzer.pixel_size = pixel_size
        analyzer.time_interval = time_interval

        st.subheader("ROI Definition")
        roi_options = ["Automated"]
        if import_imagej_roi is not None:
            roi_options.append("Import from ImageJ")
        roi_method = st.radio("ROI Definition Method", roi_options, key="ia_roi_method")

        if roi_method == "Automated":
            if st.button("🎯 Detect Bleach & Define ROIs", key="ia_detect_rois"):
                bleach_frame, bleach_coords = analyzer.detect_bleach_event(method=detection_method)
                if bleach_frame and bleach_coords:
                    st.success(f"✅ Bleach detected at frame {bleach_frame}, coordinates {bleach_coords}")
                    analyzer.define_rois(bleach_coords, bleach_radius)
                    st.info(f"📍 Defined {len(analyzer.rois)} ROIs automatically")
                else:
                    st.error("❌ Could not automatically detect bleach event.")
        
        elif roi_method == "Import from ImageJ" and import_imagej_roi is not None:
            uploaded_rois = st.file_uploader("Upload ImageJ ROI files (.roi)", type=['roi'], accept_multiple_files=True, key="ia_roi_uploader")
            if uploaded_rois:
                analyzer.rois = {}
                for roi_file in uploaded_rois:
                    try:
                        roi_info = import_imagej_roi(roi_file.getvalue())
                        roi_type = st.selectbox(f"Assign type for '{roi_info['name']}'", ['bleach_spot', 'reference', 'background'], key=f"ia_roi_type_{roi_file.name}")
                        analyzer.add_roi_from_import(roi_info, roi_type)
                    except Exception as e:
                        st.error(f"Error parsing ROI file {roi_file.name}: {e}")

        if analyzer.rois:
            st.write(f"**Current ROIs:** {len(analyzer.rois)}")
            if st.button("📈 Extract Intensity Profiles", key="ia_extract_profiles"):
                with st.spinner("Extracting intensity profiles..."):
                    df = analyzer.extract_intensity_profiles(enable_stabilization=enable_tracking)
                    st.session_state.image_analysis_df = df

                    st.subheader("📊 Extracted Intensity Data")
                    st.dataframe(df.head(10))

                    fig, ax = plt.subplots(figsize=(10, 6))
                    for roi_name in ['ROI1', 'ROI2', 'ROI3']:
                        ax.plot(df['time'], df[roi_name], label=roi_name)
                    if analyzer.bleach_frame:
                         ax.axvline(x=df['time'].iloc[analyzer.bleach_frame], color='orange', linestyle='--', label='Bleach Event')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Intensity')
                    ax.set_title('FRAP Intensity Profiles')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

            if st.session_state.image_analysis_df is not None:
                st.subheader("💾 Export and Add to Analysis")
                df = st.session_state.image_analysis_df

                col1, col2 = st.columns(2)
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Intensity Data (CSV)",
                        data=csv_data,
                        file_name=f"frap_intensities_{analyzer.file_name.split('.')[0]}.csv",
                        mime="text/csv",
                        key="ia_download_csv"
                    )
                with col2:
                    if st.button("➕ Add to Data Manager", key="ia_add_to_dm"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as tmp:
                            df.to_csv(tmp.name, index=False)
                            tmp_path = tmp.name

                        file_name = f"{os.path.splitext(analyzer.file_name)[0]}.csv"
                        if dm.load_file(tmp_path, file_name, settings=st.session_state.settings):
                            st.success(f"Successfully added '{file_name}' to the data manager.")
                        else:
                            st.error("Failed to add file to data manager.")

                        os.remove(tmp_path)
                        st.rerun()
    
    else:
        st.info("👆 Upload a TIFF image stack to begin analysis")
        
        with st.expander("📋 Supported Image Formats"):
            st.write("""
            - TIFF/TIF files (single and multi-page)
            - Image sequences in directories
            """)