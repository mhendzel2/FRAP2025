"""
Microirradiation Image Analysis Module
Enhanced image processing capabilities for laser microirradiation experiments including:
- Damage site detection and tracking
- ROI expansion measurement (chromatin decondensation)  
- Adaptive mask generation for expanding regions
- Combined microirradiation + photobleaching analysis
"""

import numpy as np
import pandas as pd
import cv2
import tifffile
from skimage import filters, measure, morphology, segmentation
from skimage.restoration import rolling_ball
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple, List, Dict, Optional, Union
import os
from pathlib import Path
from dataclasses import dataclass

# Import base functionality from existing FRAP image analysis
from frap_image_analysis import FRAPImageAnalyzer
from microirradiation_core import MicroirradiationResult, generate_adaptive_mask

@dataclass
class ROIExpansionData:
    """Container for ROI expansion tracking data"""
    time_points: np.ndarray
    areas: np.ndarray
    perimeters: np.ndarray
    centroids: List[Tuple[float, float]]
    expansion_factors: np.ndarray
    masks: List[np.ndarray]


class MicroirradiationImageAnalyzer(FRAPImageAnalyzer):
    """
    Enhanced image analyzer for microirradiation experiments
    Extends FRAPImageAnalyzer with microirradiation-specific functionality
    """
    
    def __init__(self):
        """Initialize with microirradiation-specific parameters"""
        super().__init__()
        self.damage_frame = None
        self.damage_coordinates = None
        self.initial_damage_radius = 2.0  # micrometers
        self.roi_expansion_data = None
        self.adaptive_masks = []
        
        # For combined experiments
        self.has_photobleaching = False
        self.bleach_frame = None
        self.bleach_coordinates = None
        
    def detect_damage_site(self, 
                          frame_range: Tuple[int, int] = (0, 5),
                          detection_method: str = 'intensity_change',
                          threshold_factor: float = 2.0) -> Optional[Tuple[int, int]]:
        """
        Automatically detect laser microirradiation damage site
        
        Parameters:
        -----------
        frame_range : tuple
            (start_frame, end_frame) to look for damage
        detection_method : str
            Method for detection ('intensity_change', 'edge_detection', 'manual')
        threshold_factor : float
            Factor for thresholding intensity changes
            
        Returns:
        --------
        tuple or None
            (x, y) coordinates of damage site, or None if not detected
        """
        if self.image_stack is None:
            st.error("No image stack loaded")
            return None
            
        start_frame, end_frame = frame_range
        end_frame = min(end_frame, self.image_stack.shape[0])
        
        if detection_method == 'intensity_change':
            # Look for sudden intensity changes between frames
            if start_frame + 1 >= end_frame:
                st.error("Need at least 2 frames for intensity change detection")
                return None
                
            # Calculate frame differences
            for frame_idx in range(start_frame, end_frame - 1):
                frame1 = self.image_stack[frame_idx].astype(np.float32)
                frame2 = self.image_stack[frame_idx + 1].astype(np.float32)
                
                # Calculate absolute difference
                diff = np.abs(frame2 - frame1)
                
                # Find regions of high change
                threshold = np.mean(diff) + threshold_factor * np.std(diff)
                high_change = diff > threshold
                
                # Find largest connected component
                labeled = measure.label(high_change)
                if labeled.max() > 0:
                    props = measure.regionprops(labeled)
                    # Get largest region
                    largest_region = max(props, key=lambda x: x.area)
                    
                    # Get centroid
                    centroid = largest_region.centroid
                    self.damage_coordinates = (int(centroid[1]), int(centroid[0]))  # (x, y)
                    self.damage_frame = frame_idx + 1  # Damage occurs between frames
                    
                    return self.damage_coordinates
                    
        elif detection_method == 'edge_detection':
            # Use edge detection to find new structures
            for frame_idx in range(start_frame, end_frame):
                frame = self.image_stack[frame_idx]
                edges = filters.sobel(frame)
                
                # Find peaks in edge map
                peaks = peak_local_max(edges, min_distance=10, threshold_abs=np.mean(edges) + threshold_factor * np.std(edges))
                
                if len(peaks) > 0:
                    # Take the most prominent peak
                    peak_idx = np.argmax(edges[peaks[:, 0], peaks[:, 1]])
                    self.damage_coordinates = (peaks[peak_idx, 1], peaks[peak_idx, 0])  # (x, y)
                    self.damage_frame = frame_idx
                    
                    return self.damage_coordinates
                    
        elif detection_method == 'manual':
            # Manual selection through UI
            st.info("Please manually select the damage site using the interactive interface below")
            return None
            
        st.warning("Damage site not detected automatically. Try manual selection or adjust parameters.")
        return None
    
    def track_roi_expansion(self, 
                           initial_radius: float = None,
                           method: str = 'threshold_based',
                           expansion_threshold: float = 0.1) -> ROIExpansionData:
        """
        Track ROI expansion over time (chromatin decondensation)
        
        Parameters:
        -----------
        initial_radius : float
            Initial damage ROI radius in pixels
        method : str
            Method for tracking ('threshold_based', 'edge_based', 'watershed')
        expansion_threshold : float
            Threshold for detecting expansion
            
        Returns:
        --------
        ROIExpansionData
            Complete expansion tracking data
        """
        if self.damage_coordinates is None or self.damage_frame is None:
            st.error("Damage site must be detected/set before tracking expansion")
            return None
            
        if initial_radius is None:
            initial_radius = self.initial_damage_radius / self.pixel_size
            
        n_frames = self.image_stack.shape[0]
        areas = np.zeros(n_frames)
        perimeters = np.zeros(n_frames)
        centroids = []
        masks = []
        
        # Start tracking from damage frame
        damage_frame = self.damage_frame
        center_x, center_y = self.damage_coordinates
        
        for frame_idx in range(damage_frame, n_frames):
            frame = self.image_stack[frame_idx]
            
            if method == 'threshold_based':
                # Adaptive thresholding around damage site
                # Create region of interest around damage site
                roi_size = int(initial_radius * 3)  # 3x initial radius
                x_start = max(0, center_x - roi_size)
                x_end = min(frame.shape[1], center_x + roi_size)
                y_start = max(0, center_y - roi_size)
                y_end = min(frame.shape[0], center_y + roi_size)
                
                roi = frame[y_start:y_end, x_start:x_end]
                
                # Apply adaptive threshold
                threshold = filters.threshold_otsu(roi)
                binary = roi < threshold  # Assuming damage appears darker
                
                # Clean up small artifacts
                binary = morphology.remove_small_objects(binary, min_size=10)
                binary = morphology.binary_closing(binary, morphology.disk(2))
                
                # Find largest connected component
                labeled = measure.label(binary)
                if labeled.max() > 0:
                    props = measure.regionprops(labeled)
                    largest_region = max(props, key=lambda x: x.area)
                    
                    # Create full-size mask
                    mask = np.zeros_like(frame, dtype=bool)
                    region_coords = largest_region.coords
                    for coord in region_coords:
                        y_global = coord[0] + y_start
                        x_global = coord[1] + x_start
                        if 0 <= y_global < frame.shape[0] and 0 <= x_global < frame.shape[1]:
                            mask[y_global, x_global] = True
                    
                    # Calculate properties
                    areas[frame_idx] = largest_region.area * (self.pixel_size ** 2)  # Convert to µm²
                    perimeters[frame_idx] = largest_region.perimeter * self.pixel_size  # Convert to µm
                    
                    # Global centroid
                    centroid_global = (largest_region.centroid[1] + x_start, 
                                     largest_region.centroid[0] + y_start)
                    centroids.append(centroid_global)
                    masks.append(mask)
                    
                else:
                    # No region found, use previous values or defaults
                    if frame_idx > damage_frame:
                        areas[frame_idx] = areas[frame_idx - 1]
                        perimeters[frame_idx] = perimeters[frame_idx - 1]
                        centroids.append(centroids[-1] if centroids else (center_x, center_y))
                    else:
                        areas[frame_idx] = np.pi * (initial_radius ** 2) * (self.pixel_size ** 2)
                        perimeters[frame_idx] = 2 * np.pi * initial_radius * self.pixel_size
                        centroids.append((center_x, center_y))
                    masks.append(np.zeros_like(frame, dtype=bool))
                    
            elif method == 'edge_based':
                # Edge-based expansion tracking
                frame_smooth = filters.gaussian(frame, sigma=1.0)
                edges = filters.sobel(frame_smooth)
                
                # Find contours around damage site
                roi_size = int(initial_radius * 4)
                x_start = max(0, center_x - roi_size)
                x_end = min(frame.shape[1], center_x + roi_size)
                y_start = max(0, center_y - roi_size)
                y_end = min(frame.shape[0], center_y + roi_size)
                
                roi_edges = edges[y_start:y_end, x_start:x_end]
                
                # Find contours
                threshold = np.mean(roi_edges) + expansion_threshold * np.std(roi_edges)
                contours, _ = cv2.findContours((roi_edges > threshold).astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find contour closest to damage center
                    damage_center_roi = (center_x - x_start, center_y - y_start)
                    best_contour = None
                    min_distance = float('inf')
                    
                    for contour in contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            distance = np.sqrt((cx - damage_center_roi[0])**2 + (cy - damage_center_roi[1])**2)
                            if distance < min_distance:
                                min_distance = distance
                                best_contour = contour
                    
                    if best_contour is not None:
                        area = cv2.contourArea(best_contour) * (self.pixel_size ** 2)
                        perimeter = cv2.arcLength(best_contour, True) * self.pixel_size
                        
                        areas[frame_idx] = area
                        perimeters[frame_idx] = perimeter
                        
                        # Create mask from contour
                        mask = np.zeros_like(frame, dtype=bool)
                        contour_global = best_contour.copy()
                        contour_global[:, :, 0] += x_start
                        contour_global[:, :, 1] += y_start
                        cv2.fillPoly(mask, [contour_global], True)
                        masks.append(mask)
                        
                        # Centroid
                        M = cv2.moments(best_contour)
                        cx_global = int(M["m10"] / M["m00"]) + x_start
                        cy_global = int(M["m01"] / M["m00"]) + y_start
                        centroids.append((cx_global, cy_global))
                    else:
                        # Default values
                        areas[frame_idx] = np.pi * (initial_radius ** 2) * (self.pixel_size ** 2)
                        perimeters[frame_idx] = 2 * np.pi * initial_radius * self.pixel_size
                        centroids.append((center_x, center_y))
                        masks.append(np.zeros_like(frame, dtype=bool))
                else:
                    # Default values
                    areas[frame_idx] = np.pi * (initial_radius ** 2) * (self.pixel_size ** 2)
                    perimeters[frame_idx] = 2 * np.pi * initial_radius * self.pixel_size
                    centroids.append((center_x, center_y))
                    masks.append(np.zeros_like(frame, dtype=bool))
        
        # Calculate expansion factors
        initial_area = areas[damage_frame] if areas[damage_frame] > 0 else np.pi * (initial_radius ** 2) * (self.pixel_size ** 2)
        expansion_factors = areas / initial_area
        
        # Create time points
        time_points = self.time_points[damage_frame:] if self.time_points is not None else np.arange(n_frames - damage_frame) * self.time_interval
        
        self.roi_expansion_data = ROIExpansionData(
            time_points=time_points,
            areas=areas[damage_frame:],
            perimeters=perimeters[damage_frame:],
            centroids=centroids,
            expansion_factors=expansion_factors[damage_frame:],
            masks=masks
        )
        
        return self.roi_expansion_data
    
    def generate_adaptive_masks(self, 
                               expansion_data: ROIExpansionData = None) -> List[np.ndarray]:
        """
        Generate adaptive masks based on ROI expansion for accurate intensity measurements
        
        Parameters:
        -----------
        expansion_data : ROIExpansionData
            ROI expansion tracking data
            
        Returns:
        --------
        list
            List of adaptive masks for each frame
        """
        if expansion_data is None:
            expansion_data = self.roi_expansion_data
            
        if expansion_data is None:
            st.error("ROI expansion data not available. Run track_roi_expansion first.")
            return []
            
        # Use the masks from expansion tracking
        self.adaptive_masks = expansion_data.masks
        return self.adaptive_masks
    
    def extract_adaptive_intensities(self, 
                                   masks: List[np.ndarray] = None,
                                   background_correction: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract intensities using adaptive masks
        
        Parameters:
        -----------
        masks : list
            List of adaptive masks for each frame
        background_correction : bool
            Whether to apply background correction
            
        Returns:
        --------
        dict
            Dictionary with time series data
        """
        if masks is None:
            masks = self.adaptive_masks
            
        if not masks or self.image_stack is None:
            st.error("Masks and image stack required for intensity extraction")
            return {}
            
        n_frames = len(masks)
        damage_intensities = np.zeros(n_frames)
        background_intensities = np.zeros(n_frames)
        
        start_frame = self.damage_frame if self.damage_frame is not None else 0
        
        for i, mask in enumerate(masks):
            frame_idx = start_frame + i
            if frame_idx >= self.image_stack.shape[0]:
                break
                
            frame = self.image_stack[frame_idx]
            
            # Extract damage ROI intensity
            if np.any(mask):
                damage_intensities[i] = np.mean(frame[mask])
            else:
                damage_intensities[i] = np.nan
                
            # Extract background intensity (annular region around damage)
            if background_correction and np.any(mask):
                # Create background mask (annular region)
                kernel = morphology.disk(5)
                expanded_mask = morphology.binary_dilation(mask, kernel)
                background_mask = expanded_mask & ~mask
                
                if np.any(background_mask):
                    background_intensities[i] = np.mean(frame[background_mask])
                else:
                    background_intensities[i] = np.mean(frame)  # Use whole frame as fallback
            else:
                background_intensities[i] = 0
        
        # Create time points
        time_points = self.time_points[start_frame:start_frame+n_frames] if self.time_points is not None else np.arange(n_frames) * self.time_interval
        
        return {
            'time': time_points,
            'damage_intensity': damage_intensities,
            'background_intensity': background_intensities,
            'corrected_intensity': damage_intensities - background_intensities if background_correction else damage_intensities
        }
    
    def detect_photobleaching(self, 
                            intensity_data: Dict[str, np.ndarray],
                            detection_threshold: float = 0.3) -> Optional[int]:
        """
        Detect photobleaching event in intensity time series
        
        Parameters:
        -----------
        intensity_data : dict
            Intensity time series data
        detection_threshold : float
            Fractional drop threshold for bleaching detection
            
        Returns:
        --------
        int or None
            Frame index of photobleaching event, or None if not detected
        """
        if 'corrected_intensity' not in intensity_data:
            st.error("Corrected intensity data required for bleaching detection")
            return None
            
        intensity = intensity_data['corrected_intensity']
        
        # Look for sudden drops in intensity
        for i in range(1, len(intensity)):
            if not np.isnan(intensity[i-1]) and not np.isnan(intensity[i]):
                drop_fraction = (intensity[i-1] - intensity[i]) / intensity[i-1]
                if drop_fraction > detection_threshold:
                    self.has_photobleaching = True
                    self.bleach_frame = self.damage_frame + i if self.damage_frame else i
                    return self.bleach_frame
                    
        return None
    
    def create_microirradiation_interface(self, data_manager):
        """Create Streamlit interface for microirradiation analysis"""
        st.header("🔬 Microirradiation Image Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload image stack (TIFF format)",
            type=['tif', 'tiff'],
            help="Upload a time-lapse image stack of microirradiation experiment"
        )
        
        if uploaded_file is not None:
            # Load image stack
            with st.spinner("Loading image stack..."):
                # Save uploaded file temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                success = self.load_image_stack(temp_path)
                
            if success:
                st.success(f"Loaded image stack: {self.image_stack.shape}")
                
                # Display first frame
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("First Frame")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(self.image_stack[0], cmap='gray')
                    ax.set_title("Frame 0")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                with col2:
                    st.subheader("Analysis Parameters")
                    
                    # Calibration
                    self.pixel_size = st.number_input(
                        "Pixel size (µm)", 
                        min_value=0.01, max_value=1.0, 
                        value=0.1, step=0.01,
                        help="Size of one pixel in micrometers"
                    )
                    
                    self.time_interval = st.number_input(
                        "Time interval (s)", 
                        min_value=0.1, max_value=60.0, 
                        value=1.0, step=0.1,
                        help="Time between frames in seconds"
                    )
                    
                    # Generate time points
                    self.time_points = np.arange(self.image_stack.shape[0]) * self.time_interval
                
                # Damage site detection
                st.subheader("🎯 Damage Site Detection")
                
                detection_method = st.selectbox(
                    "Detection method",
                    ['intensity_change', 'edge_detection', 'manual'],
                    help="Method for automatically detecting damage site"
                )
                
                if detection_method != 'manual':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        start_frame = st.number_input("Start frame", min_value=0, max_value=self.image_stack.shape[0]-2, value=0)
                    with col2:
                        end_frame = st.number_input("End frame", min_value=start_frame+1, max_value=self.image_stack.shape[0]-1, value=min(5, self.image_stack.shape[0]-1))
                    with col3:
                        threshold_factor = st.number_input("Threshold factor", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
                    
                    if st.button("🔍 Detect Damage Site"):
                        with st.spinner("Detecting damage site..."):
                            coords = self.detect_damage_site(
                                frame_range=(start_frame, end_frame),
                                detection_method=detection_method,
                                threshold_factor=threshold_factor
                            )
                            
                        if coords:
                            st.success(f"Damage site detected at: {coords}")
                            st.session_state['damage_coords'] = coords
                            st.session_state['damage_frame'] = self.damage_frame
                        else:
                            st.warning("Damage site not detected. Try manual selection.")
                
                else:
                    # Manual selection interface
                    st.info("Click on the image below to manually select the damage site")
                    
                    frame_to_show = st.slider("Frame for selection", 0, self.image_stack.shape[0]-1, 
                                            value=self.damage_frame if self.damage_frame else 0)
                    
                    # Create interactive plot for manual selection
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(self.image_stack[frame_to_show], cmap='gray')
                    ax.set_title(f"Frame {frame_to_show} - Click to select damage site")
                    
                    # Show existing selection if available
                    if 'damage_coords' in st.session_state:
                        coords = st.session_state['damage_coords']
                        ax.plot(coords[0], coords[1], 'r+', markersize=15, markeredgewidth=3)
                        
                    st.pyplot(fig)
                    
                    # Manual coordinate input
                    col1, col2 = st.columns(2)
                    with col1:
                        manual_x = st.number_input("X coordinate", min_value=0, max_value=self.image_stack.shape[2]-1, value=self.image_stack.shape[2]//2)
                    with col2:
                        manual_y = st.number_input("Y coordinate", min_value=0, max_value=self.image_stack.shape[1]-1, value=self.image_stack.shape[1]//2)
                    
                    if st.button("Set Damage Site"):
                        self.damage_coordinates = (manual_x, manual_y)
                        self.damage_frame = frame_to_show
                        st.session_state['damage_coords'] = (manual_x, manual_y)
                        st.session_state['damage_frame'] = frame_to_show
                        st.success(f"Damage site set at: ({manual_x}, {manual_y})")
                
                # ROI expansion analysis
                if 'damage_coords' in st.session_state:
                    st.subheader("📏 ROI Expansion Analysis")
                    
                    self.damage_coordinates = st.session_state['damage_coords']
                    self.damage_frame = st.session_state['damage_frame']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        initial_radius = st.number_input(
                            "Initial damage radius (µm)", 
                            min_value=0.5, max_value=10.0, 
                            value=2.0, step=0.1
                        )
                        self.initial_damage_radius = initial_radius
                        
                    with col2:
                        expansion_method = st.selectbox(
                            "Expansion tracking method",
                            ['threshold_based', 'edge_based'],
                            help="Method for tracking ROI expansion over time"
                        )
                    
                    if st.button("📊 Analyze ROI Expansion"):
                        with st.spinner("Tracking ROI expansion..."):
                            expansion_data = self.track_roi_expansion(
                                initial_radius=initial_radius/self.pixel_size,
                                method=expansion_method
                            )
                            
                        if expansion_data:
                            st.success("ROI expansion analysis complete!")
                            
                            # Plot expansion results
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                            
                            # Area over time
                            ax1.plot(expansion_data.time_points, expansion_data.areas)
                            ax1.set_xlabel('Time (s)')
                            ax1.set_ylabel('ROI Area (µm²)')
                            ax1.set_title('ROI Area Expansion')
                            ax1.grid(True)
                            
                            # Expansion factor over time
                            ax2.plot(expansion_data.time_points, expansion_data.expansion_factors)
                            ax2.set_xlabel('Time (s)')
                            ax2.set_ylabel('Expansion Factor')
                            ax2.set_title('ROI Expansion Factor')
                            ax2.grid(True)
                            
                            st.pyplot(fig)
                            
                            # Store results
                            st.session_state['expansion_data'] = expansion_data
                    
                    # Intensity analysis with adaptive masks
                    if 'expansion_data' in st.session_state:
                        st.subheader("💡 Adaptive Intensity Analysis")
                        
                        if st.button("📈 Extract Intensities"):
                            with st.spinner("Extracting intensities with adaptive masks..."):
                                expansion_data = st.session_state['expansion_data']
                                adaptive_masks = self.generate_adaptive_masks(expansion_data)
                                intensity_data = self.extract_adaptive_intensities(
                                    masks=adaptive_masks,
                                    background_correction=True
                                )
                                
                            if intensity_data:
                                st.success("Intensity extraction complete!")
                                
                                # Plot intensity time series
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(intensity_data['time'], intensity_data['damage_intensity'], 
                                       label='Damage ROI', linewidth=2)
                                ax.plot(intensity_data['time'], intensity_data['background_intensity'], 
                                       label='Background', alpha=0.7)
                                ax.plot(intensity_data['time'], intensity_data['corrected_intensity'], 
                                       label='Corrected (Damage - Background)', linewidth=2)
                                
                                # Mark damage event
                                if self.damage_frame:
                                    damage_time = self.damage_frame * self.time_interval
                                    ax.axvline(x=damage_time, color='red', linestyle='--', 
                                             label='Microirradiation')
                                
                                ax.set_xlabel('Time (s)')
                                ax.set_ylabel('Intensity (AU)')
                                ax.set_title('Protein Recruitment Kinetics')
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                
                                # Store intensity data
                                st.session_state['intensity_data'] = intensity_data
                                
                                # Check for photobleaching
                                bleach_frame = self.detect_photobleaching(intensity_data)
                                if bleach_frame:
                                    st.info(f"🔥 Photobleaching detected at frame {bleach_frame}")
                                    st.session_state['bleach_frame'] = bleach_frame
                                    st.session_state['has_photobleaching'] = True
                
                # Analysis and export
                if 'intensity_data' in st.session_state and 'expansion_data' in st.session_state:
                    st.subheader("📊 Analysis Results")
                    
                    # Perform comprehensive analysis
                    from microirradiation_core import analyze_recruitment_kinetics, analyze_roi_expansion, analyze_combined_experiment
                    
                    intensity_data = st.session_state['intensity_data']
                    expansion_data = st.session_state['expansion_data']
                    
                    # Recruitment kinetics analysis
                    recruitment_results = analyze_recruitment_kinetics(
                        intensity_data['time'],
                        intensity_data['corrected_intensity'], 
                        damage_frame=0  # Already adjusted for damage start
                    )
                    
                    # ROI expansion analysis
                    expansion_results = analyze_roi_expansion(
                        expansion_data.time_points,
                        expansion_data.areas,
                        damage_frame=0  # Already adjusted
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Recruitment Kinetics**")
                        if recruitment_results['best_fit']:
                            best_fit = recruitment_results['best_fit']
                            st.write(f"- Best model: {recruitment_results['best_model']}")
                            st.write(f"- Recruitment rate: {best_fit.get('rate', 'N/A'):.4f} s⁻¹")
                            st.write(f"- Half-time: {best_fit.get('half_time', 'N/A'):.2f} s")
                            st.write(f"- Amplitude: {best_fit.get('amplitude', 'N/A'):.2f}")
                            st.write(f"- R²: {best_fit.get('r_squared', 'N/A'):.4f}")
                        else:
                            st.write("No successful fit")
                    
                    with col2:
                        st.write("**ROI Expansion**")
                        if expansion_results['best_fit']:
                            best_fit = expansion_results['best_fit']
                            st.write(f"- Best model: {expansion_results['best_model']}")
                            st.write(f"- Expansion rate: {best_fit.get('rate', 'N/A'):.4f}")
                            st.write(f"- Initial area: {best_fit.get('initial_size', 'N/A'):.2f} µm²")
                            st.write(f"- Max expansion: {best_fit.get('max_expansion', 'N/A'):.2f} µm²")
                            st.write(f"- R²: {best_fit.get('r_squared', 'N/A'):.4f}")
                        else:
                            st.write("No successful fit")
                    
                    # Export results
                    if st.button("💾 Export Results"):
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            'Time_s': intensity_data['time'],
                            'Damage_Intensity': intensity_data['damage_intensity'],
                            'Background_Intensity': intensity_data['background_intensity'],
                            'Corrected_Intensity': intensity_data['corrected_intensity'],
                            'ROI_Area_um2': np.concatenate([[np.nan] * (len(intensity_data['time']) - len(expansion_data.areas)), expansion_data.areas]),
                            'Expansion_Factor': np.concatenate([[np.nan] * (len(intensity_data['time']) - len(expansion_data.expansion_factors)), expansion_data.expansion_factors])
                        })
                        
                        # Convert to CSV
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name=f"microirradiation_results_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
            
            else:
                st.error("Failed to load image stack. Please check the file format.")