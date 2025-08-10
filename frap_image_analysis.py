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
    
    def extract_intensity_profiles(self, apply_background_correction: bool = True) -> pd.DataFrame:
        """
        Extract intensity profiles from defined ROIs
        
        Parameters:
        -----------
        apply_background_correction : bool
            Whether to apply background correction
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time points and intensity values
        """
        if self.image_stack is None or not self.rois:
            raise ValueError("No image stack or ROIs defined")
        
        n_frames = len(self.image_stack)
        
        # Extract intensities for each ROI
        roi1_intensities = np.zeros(n_frames)
        roi2_intensities = np.zeros(n_frames)
        roi3_intensities = np.zeros(n_frames)
        
        dynamic_tracking = self.tracked_centers is not None and len(self.tracked_centers) == n_frames
        height, width = self.image_stack.shape[1:]
        if dynamic_tracking:
            # Precompute static background mask
            bg_mask = self.rois['ROI3']['mask']
            for t in range(n_frames):
                frame = self.image_stack[t].astype(float)
                if apply_background_correction:
                    frame = self._apply_background_correction(frame)
                cx, cy = self.tracked_centers[t]
                # Rebuild ROI1/ROI2 masks around tracked center
                bleach_radius = self.rois['ROI1']['radius']
                ref_inner = self.rois['ROI2']['inner_radius']
                ref_outer = self.rois['ROI2']['outer_radius']
                y_grid, x_grid = np.ogrid[:height, :width]
                roi1_mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= bleach_radius**2
                annulus_mask = ((x_grid - cx)**2 + (y_grid - cy)**2 >= ref_inner**2) & ((x_grid - cx)**2 + (y_grid - cy)**2 <= ref_outer**2)
                roi1_intensities[t] = np.mean(frame[roi1_mask])
                roi2_intensities[t] = np.mean(frame[annulus_mask])
                roi3_intensities[t] = np.mean(frame[bg_mask])
        else:
            for t in range(n_frames):
                frame = self.image_stack[t].astype(float)
                if apply_background_correction:
                    frame = self._apply_background_correction(frame)
                roi1_intensities[t] = np.mean(frame[self.rois['ROI1']['mask']])
                roi2_intensities[t] = np.mean(frame[self.rois['ROI2']['mask']])
                roi3_intensities[t] = np.mean(frame[self.rois['ROI3']['mask']])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Time': self.time_points[:n_frames],
            'ROI1': roi1_intensities,
            'ROI2': roi2_intensities,
            'ROI3': roi3_intensities
        })
        
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

def create_image_analysis_interface():
    """Create Streamlit interface for image analysis"""
    st.header("🔬 FRAP Image Analysis")
    st.write("Direct analysis of raw microscopy images with automated ROI detection")
    
    analyzer = FRAPImageAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload FRAP Image Stack",
        type=['tif', 'tiff'],
        help="Upload TIFF files or image sequences"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load image stack
            if analyzer.load_image_stack(temp_path):
                st.success(f"✅ Loaded image stack: {analyzer.image_stack.shape}")
                
                # Display summary
                summary = analyzer.create_analysis_summary()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Frames", summary['image_info']['n_frames'])
                with col2:
                    st.metric("Image Size", f"{summary['image_info']['image_shape'][1]}×{summary['image_info']['image_shape'][0]}")
                with col3:
                    st.metric("Pixel Size", f"{summary['image_info']['pixel_size_um']:.2f} μm")
                
                # Parameters
                st.subheader("Analysis Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    pixel_size = st.number_input("Pixel Size (μm)", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
                    time_interval = st.number_input("Time Interval (s)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
                
                with col2:
                    bleach_radius = st.number_input("Bleach Spot Radius (pixels)", value=10, min_value=3, max_value=50)
                    detection_method = st.selectbox("Bleach Detection", ["intensity_drop", "gradient"])
                
                analyzer.pixel_size = pixel_size
                analyzer.time_interval = time_interval
                
                # Detect bleach event
                if st.button("🎯 Detect Bleach Event"):
                    try:
                        bleach_frame, bleach_coords = analyzer.detect_bleach_event(method=detection_method)
                        
                        if bleach_frame is not None and bleach_coords is not None:
                            st.success(f"✅ Bleach detected at frame {bleach_frame}, coordinates {bleach_coords}")
                            
                            # Define ROIs
                            rois = analyzer.define_rois(bleach_coords, bleach_radius)
                            st.info(f"📍 Defined {len(rois)} ROIs automatically")
                            
                            # Extract intensity profiles
                            with st.spinner("Extracting intensity profiles..."):
                                df = analyzer.extract_intensity_profiles()
                                
                                st.subheader("📊 Extracted Intensity Data")
                                st.dataframe(df.head(10))
                                
                                # Plot intensity profiles
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(df['Time'], df['ROI1'], 'r-', label='ROI1 (Bleach)', linewidth=2)
                                ax.plot(df['Time'], df['ROI2'], 'g-', label='ROI2 (Reference)', linewidth=2)
                                ax.plot(df['Time'], df['ROI3'], 'b-', label='ROI3 (Background)', linewidth=2)
                                ax.axvline(x=df['Time'].iloc[bleach_frame], color='orange', linestyle='--', label='Bleach Event')
                                ax.set_xlabel('Time (s)')
                                ax.set_ylabel('Intensity')
                                ax.set_title('FRAP Intensity Profiles')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                
                                # PSF analysis
                                st.subheader("🔍 PSF Analysis")
                                psf_params = analyzer.estimate_psf_parameters()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Effective Radius", f"{psf_params['effective_radius_um']:.2f} μm")
                                with col2:
                                    st.metric("Radius (pixels)", f"{psf_params['effective_radius_pixels']:.1f}")
                                with col3:
                                    st.metric("Pixel Size", f"{psf_params['pixel_size_um']:.3f} μm")
                                
                                # Download processed data
                                st.subheader("💾 Export Data")
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Intensity Data (CSV)",
                                    data=csv_data,
                                    file_name=f"frap_intensities_{uploaded_file.name.split('.')[0]}.csv",
                                    mime="text/csv"
                                )
                                
                                # Store in session state for further analysis
                                if 'image_analysis_data' not in st.session_state:
                                    st.session_state.image_analysis_data = {}
                                
                                st.session_state.image_analysis_data[uploaded_file.name] = {
                                    'dataframe': df,
                                    'psf_params': psf_params,
                                    'analysis_summary': summary
                                }
                                
                                st.success("✅ Image analysis complete! Data ready for kinetic fitting.")
                        
                        else:
                            st.error("❌ Could not automatically detect bleach event. Try manual ROI selection.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
            
            else:
                st.error("❌ Failed to load image stack")
        
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
    
    else:
        st.info("👆 Upload a TIFF image stack to begin analysis")
        
        # Show supported formats
        with st.expander("📋 Supported Image Formats"):
            st.write("""
            **Currently Supported:**
            - TIFF/TIF files (single and multi-page)
            - Image sequences in directories
            
            **Future Enhancement (requires additional packages):**
            - Zeiss CZI files (install: `pip install czifile`)
            - Leica LIF files (install: `pip install readlif`)
            - OME-TIFF and other formats (Bio-Formats integration)
            
            **Recommended Workflow:**
            1. Export your microscopy data as TIFF stacks
            2. Ensure consistent time intervals
            3. Include metadata when possible
            """)