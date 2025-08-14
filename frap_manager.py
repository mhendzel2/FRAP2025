import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
import tempfile
import shutil
import logging
from frap_core_corrected import FRAPAnalysisCore
from frap_image_analysis import FRAPImageAnalyzer

logger = logging.getLogger(__name__)

def validate_analysis_results(features: dict) -> dict:
    """
    Validate and correct analysis results to ensure they are physically reasonable.

    Parameters:
    -----------
    features : dict
        Dictionary containing analysis features

    Returns:
    --------
    dict
        Validated and corrected features
    """
    if not features:
        return features

    # Validate mobile fraction (should be 0-100%)
    mobile_fraction = features.get('mobile_fraction', np.nan)
    if np.isfinite(mobile_fraction):
        if mobile_fraction < 0:
            logger.warning(f"Mobile fraction {mobile_fraction:.1f}% is negative, setting to 0%")
            features['mobile_fraction'] = 0.0
        elif mobile_fraction > 100:
            logger.warning(f"Mobile fraction {mobile_fraction:.1f}% exceeds 100%, capping at 100%")
            features['mobile_fraction'] = 100.0

    # Ensure immobile fraction = 100% - mobile fraction
    if np.isfinite(features.get('mobile_fraction', np.nan)):
        features['immobile_fraction'] = 100.0 - features['mobile_fraction']
    else:
        features['immobile_fraction'] = np.nan

    # Validate rate constants (should be positive)
    for key in features:
        if 'rate_constant' in key and np.isfinite(features[key]):
            if features[key] <= 0:
                logger.warning(f"Rate constant {key} = {features[key]:.6f} is non-positive, setting to NaN")
                features[key] = np.nan

    # Validate half-times (should be positive)
    for key in features:
        if 'half_time' in key and np.isfinite(features[key]):
            if features[key] <= 0:
                logger.warning(f"Half-time {key} = {features[key]:.6f} is non-positive, setting to NaN")
                features[key] = np.nan

    # Validate proportions (should sum to ~100% for mobile pool)
    mobile_props = [features.get(f'proportion_of_mobile_{comp}', 0)
                   for comp in ['fast', 'medium', 'slow']]
    mobile_props = [p for p in mobile_props if np.isfinite(p)]

    if mobile_props and abs(sum(mobile_props) - 100.0) > 5.0:  # Allow 5% tolerance
        logger.warning(f"Mobile pool proportions sum to {sum(mobile_props):.1f}%, not ~100%")

    return features

class FRAPDataManager:
    def __init__(self):
        self.files,self.groups = {},{}

    def load_file(self,file_path,file_name):
        try:
            # Extract original extension before the hash suffix
            original_path = file_path
            if '_' in file_path and any(ext in file_path for ext in ['.xls_', '.xlsx_', '.csv_']):
                # Find the original extension and create a temporary file with correct extension
                import tempfile
                import shutil
                if '.xlsx_' in file_path:
                    temp_path = tempfile.mktemp(suffix='.xlsx')
                elif '.xls_' in file_path:
                    temp_path = tempfile.mktemp(suffix='.xls')
                elif '.csv_' in file_path:
                    temp_path = tempfile.mktemp(suffix='.csv')
                else:
                    temp_path = file_path

                if temp_path != file_path:
                    shutil.copy2(file_path, temp_path)
                    file_path = temp_path

            processed_df = FRAPAnalysisCore.preprocess(FRAPAnalysisCore.load_data(file_path))
            if 'normalized' in processed_df.columns and not processed_df['normalized'].isnull().all():
                time,intensity = processed_df['time'].values,processed_df['normalized'].values

                # Validate the normalized data
                if np.any(intensity < 0):
                    logger.warning(f"Negative intensities found in normalized data for {file_name}")
                    # Shift to ensure all values are non-negative
                    intensity = intensity - np.min(intensity)

                # Ensure proper normalization (pre-bleach should be ~1.0)
                bleach_idx = np.argmin(intensity)
                if bleach_idx > 0:
                    pre_bleach_mean = np.mean(intensity[:bleach_idx])
                    if not np.isclose(pre_bleach_mean, 1.0, rtol=0.1):
                        logger.warning(f"Pre-bleach intensity not normalized to ~1.0 (got {pre_bleach_mean:.3f}) for {file_name}")
                        # Re-normalize if necessary
                        if pre_bleach_mean > 0:
                            intensity = intensity / pre_bleach_mean

                fits = FRAPAnalysisCore.fit_all_models(time,intensity)
                best_fit = FRAPAnalysisCore.select_best_fit(fits,st.session_state.settings['default_criterion'])

                if best_fit:
                    params = FRAPAnalysisCore.extract_clustering_features(best_fit)
                    # Validate the analysis results
                    params = validate_analysis_results(params)
                else:
                    params = None
                    logger.error(f"No valid fit found for {file_name}")

                self.files[file_path]={
                    'name':file_name,'data':processed_df,'time':time,'intensity':intensity,
                    'fits':fits,'best_fit':best_fit,'features':params
                }
                logger.info(f"Loaded: {file_name}")
                return True
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            logger.error(f"Detailed error for {file_name}: {e}", exc_info=True)
            return False

    def create_group(self,name):
        if name not in self.groups:
            self.groups[name]={'name':name,'files':[],'features_df':None}

    def update_group_analysis(self,name,excluded_files=None):
        if name not in self.groups: return
        group=self.groups[name]
        features_list=[]
        for fp in group['files']:
            if fp not in (excluded_files or []) and fp in self.files and self.files[fp]['features']:
                ff=self.files[fp]['features'].copy()
                ff.update({'file_path':fp,'file_name':self.files[fp]['name']})
                features_list.append(ff)
        group['features_df'] = pd.DataFrame(features_list) if features_list else pd.DataFrame()

    def add_file_to_group(self, group_name, file_path):
        """
        Adds a file to an existing group.
        """
        if group_name in self.groups and file_path in self.files:
            if file_path not in self.groups[group_name]['files']:
                self.groups[group_name]['files'].append(file_path)
                return True
        return False

    def fit_group_models(self, group_name, model='single', excluded_files=None):
        """
        Perform global simultaneous fitting for a group with shared kinetic parameters
        but individual amplitudes.

        Parameters:
        -----------
        group_name : str
            Name of the group to fit
        model : str
            Model type ('single', 'double', or 'triple')
        excluded_files : list, optional
            List of file paths to exclude from global fitting

        Returns:
        --------
        dict
            Dictionary containing global fit results
        """
        if group_name not in self.groups:
            raise KeyError(f"Group {group_name} not found.")

        group = self.groups[group_name]
        excluded_files = excluded_files or []

        # Prepare traces for global fitting
        traces = []
        file_names = []

        for file_path in group['files']:
            if file_path not in excluded_files and file_path in self.files:
                file_data = self.files[file_path]
                t, y, _ = FRAPAnalysisCore.get_post_bleach_data(
                    file_data['time'],
                    file_data['intensity']
                )
                traces.append((t, y))
                file_names.append(file_data['name'])

        if len(traces) < 2:
            raise ValueError("Need at least 2 traces for global fitting")

        try:
            # Perform global fitting using the core analysis function
            global_fit_result = FRAPAnalysisCore.fit_group_models(traces, model=model)

            # Add file names for reference
            global_fit_result['file_names'] = file_names
            global_fit_result['excluded_files'] = excluded_files

            # Store the result in the group
            if 'global_fit' not in group:
                group['global_fit'] = {}
            group['global_fit'][model] = global_fit_result

            return global_fit_result

        except Exception as e:
            logger.error(f"Error in global fitting for group {group_name}: {e}")
            return {
                'model': model,
                'success': False,
                'error': str(e)
            }

    def load_groups_from_zip_archive(self, zip_file):
        """
        Loads files from a ZIP archive containing subfolders, where each subfolder
        is treated as a new group. Gracefully handles unreadable files.
        """
        success_count = 0
        error_count = 0
        error_details = []
        groups_created = []

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with zipfile.ZipFile(io.BytesIO(zip_file.getbuffer())) as z:
                        z.extractall(temp_dir)
                        logger.info(f"Extracted ZIP archive to: {temp_dir}")
                except zipfile.BadZipFile:
                    logger.error("Invalid ZIP file format")
                    st.error("Invalid ZIP file format. Please check that the uploaded file is a valid ZIP archive.")
                    return False
                except Exception as e:
                    logger.error(f"Error extracting ZIP file: {e}")
                    st.error(f"Error extracting ZIP file: {e}")
                    return False

                # Walk through the extracted directory to find subfolders (groups)
                for root, dirs, files in os.walk(temp_dir):
                    # Process subfolders as groups
                    for group_name in dirs:
                        if group_name.startswith('__'):  # Ignore system folders like __MACOSX
                            continue

                        self.create_group(group_name)
                        groups_created.append(group_name)
                        group_path = os.path.join(root, group_name)

                        for file_in_group in os.listdir(group_path):
                            file_path_in_temp = os.path.join(group_path, file_in_group)

                            if os.path.isfile(file_path_in_temp) and not file_in_group.startswith('.'):
                                try:
                                    file_ext = os.path.splitext(file_in_group)[1].lower()
                                    file_name = os.path.basename(file_in_group)

                                    if file_ext not in ['.xls', '.xlsx', '.csv', '.tif', '.tiff']:
                                        continue

                                    with open(file_path_in_temp, 'rb') as f:
                                        file_content = f.read()
                                    if not file_content:
                                        continue

                                    content_hash = hash(file_content)

                                    if file_ext in ['.tif', '.tiff']:
                                        base_name = os.path.splitext(file_name)[0]
                                        tp = f"data/{base_name}_{content_hash}.csv"
                                    else:
                                        tp = f"data/{file_name}_{content_hash}"

                                    if tp not in self.files:
                                        os.makedirs(os.path.dirname(tp), exist_ok=True)

                                        if file_ext in ['.tif', '.tiff']:
                                            analyzer = FRAPImageAnalyzer()
                                            if not analyzer.load_image_stack(file_path_in_temp):
                                                raise ValueError("Failed to load image stack.")

                                            settings = st.session_state.settings
                                            analyzer.pixel_size = settings.get('default_pixel_size', 0.3)
                                            analyzer.time_interval = settings.get('default_time_interval', 1.0)

                                            bleach_frame, bleach_coords = analyzer.detect_bleach_event()
                                            if bleach_frame is None or bleach_coords is None:
                                                raise ValueError("Failed to detect bleach event automatically.")

                                            bleach_radius_pixels = int(settings.get('default_bleach_radius', 1.0) / analyzer.pixel_size)
                                            analyzer.define_rois(bleach_coords, bleach_radius=bleach_radius_pixels)
                                            intensity_df = analyzer.extract_intensity_profiles()
                                            intensity_df = intensity_df.rename(columns={'Time': 'time'})
                                            intensity_df.to_csv(tp, index=False)
                                        else:
                                            shutil.copy(file_path_in_temp, tp)

                                        if self.load_file(tp, file_name):
                                            self.add_file_to_group(group_name, tp)
                                            success_count += 1
                                        else:
                                            raise ValueError("Failed to load data from file.")
                                    else:
                                        self.add_file_to_group(group_name, tp)

                                except Exception as e:
                                    error_count += 1
                                    error_details.append(f"Error processing file {file_in_group} in group {group_name}: {str(e)}")
                                    logger.error(f"Error processing file {file_in_group} in group {group_name}: {str(e)}", exc_info=True)
                                    if 'tp' in locals() and tp not in self.files and os.path.exists(tp):
                                        os.remove(tp)

            # After processing all groups/files
            if success_count > 0:
                for group_name in groups_created:
                    self.update_group_analysis(group_name)
                st.success(f"Successfully loaded {success_count} files into {len(groups_created)} groups.")
                if error_count > 0:
                    st.warning(f"{error_count} files were skipped due to errors.")
                    with st.expander("View Error Details"):
                        for error in error_details:
                            st.text(error)
                return True
            else:
                st.error("No files could be processed from the ZIP archive.")
                return False

        except Exception as e:
            logger.error(f"Error processing ZIP archive with subfolders: {e}")
            st.error(f"An unexpected error occurred: {e}")
            return False

    def load_zip_archive_and_create_group(self, zip_file, group_name):
        """
        Loads files from a ZIP archive, creates a new group, and adds the files to it.
        Gracefully handles unreadable files.
        """
        success_count = 0
        error_count = 0
        error_details = []

        try:
            self.create_group(group_name)
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                file_list = z.namelist()
                for file_in_zip in file_list:
                    # Ignore directories and hidden files (like __MACOSX)
                    if not file_in_zip.endswith('/') and '__MACOSX' not in file_in_zip:
                        try:
                            # Check if file has valid extension
                            file_ext = os.path.splitext(file_in_zip)[1].lower()
                            if file_ext not in ['.xls', '.xlsx', '.csv']:
                                logger.warning(f"Skipping unsupported file type: {file_in_zip}")
                                continue

                            file_content = z.read(file_in_zip)

                            # Skip empty files
                            if len(file_content) == 0:
                                logger.warning(f"Skipping empty file: {file_in_zip}")
                                continue

                            file_name = os.path.basename(file_in_zip)

                            # Create a temporary file to use with the existing load_file logic
                            tp = f"data/{file_name}_{hash(file_content)}"
                            if tp not in self.files:
                                with open(tp, "wb") as f:
                                    f.write(file_content)

                                # Attempt to load the file
                                if self.load_file(tp, file_name):
                                    self.add_file_to_group(group_name, tp)
                                    success_count += 1
                                else:
                                    error_count += 1
                                    error_msg = f"Failed to load file: {file_in_zip}"
                                    error_details.append(error_msg)
                                    logger.error(error_msg)
                                    # Clean up failed file
                                    if os.path.exists(tp):
                                        os.remove(tp)
                            else:
                                # File already exists, just add to group
                                self.add_file_to_group(group_name, tp)
                                success_count += 1

                        except Exception as e:
                            error_count += 1
                            error_msg = f"Error processing file {file_in_zip}: {str(e)}"
                            error_details.append(error_msg)
                            logger.error(error_msg)
                            continue

            # Update group analysis only if we have successfully loaded files
            if success_count > 0:
                self.update_group_analysis(group_name)
                st.session_state.selected_group_name = group_name

                # Report results
                logger.info(f"Successfully processed {success_count} files for group {group_name}")
                if error_count > 0:
                    logger.warning(f"{error_count} files could not be processed and were skipped")
                    # Display error details in Streamlit
                    if hasattr(st, 'warning'):
                        st.warning(f"Successfully loaded {success_count} files to group '{group_name}'. "
                                 f"{error_count} files were skipped due to errors.")
                        if error_details:
                            with st.expander("⚠️ View Skipped Files Details"):
                                for error in error_details[:10]:  # Show first 10 errors
                                    st.text(f"• {error}")
                                if len(error_details) > 10:
                                    st.text(f"... and {len(error_details) - 10} more errors")
                return True
            else:
                # No files could be loaded, remove the empty group
                logger.error(f"No files could be processed for group {group_name}")
                if group_name in self.groups:
                    del self.groups[group_name]
                return False

        except Exception as e:
            logger.error(f"Error processing ZIP archive for group {group_name}: {e}")
            # Clean up group if creation failed
            if group_name in self.groups:
                del self.groups[group_name]
            return False
