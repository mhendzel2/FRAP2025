"""
FRAP Data Management Module
Handle data loading, processing, and management for single files and groups
"""

import os
import glob
import pandas as pd
import numpy as np
from frap_core import FRAPAnalysisCore
import logging

logger = logging.getLogger(__name__)

class FRAPData:
    def __init__(self):
        """
        Initialize FRAPData class to manage FRAP data files and groups
        """
        self.files = {}  # Dictionary to store loaded files
        self.groups = {}  # Dictionary to store groups of files
        self.current_file = None
        self.current_group = None
        
    def load_file(self, file_path):
        """
        Load a single FRAP data file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
            
        Returns:
        --------
        dict
            Dictionary containing the file data and metadata
        """
        try:
            # Load the data
            df = FRAPAnalysisCore.load_data(file_path)
            
            # Process the data
            processed_df = FRAPAnalysisCore.preprocess(df)
            
            # Get filename for display
            file_name = os.path.basename(file_path)
            
            # Store the data
            file_data = {
                'path': file_path,
                'name': file_name,
                'raw_data': df,
                'processed_data': processed_df,
                'time': processed_df['time'].values,
                'intensity': processed_df['normalized'].values,
                'fits': None,
                'best_fit': None,
                'features': None
            }
            
            # Perform the fits
            fits = FRAPAnalysisCore.fit_all_models(
                file_data['time'],
                file_data['intensity']
            )
            file_data['fits'] = fits
            
            # Select best fit (using AIC by default)
            if fits:
                best_fit = FRAPAnalysisCore.select_best_fit(fits, criterion='aic')
                file_data['best_fit'] = best_fit
                
                # Extract features for potential clustering
                features = FRAPAnalysisCore.extract_clustering_features(best_fit)
                file_data['features'] = features
            
            # Store in files dictionary
            self.files[file_path] = file_data
            self.current_file = file_path
            
            return file_data
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise e
    
    def load_directory(self, directory_path, file_pattern="*.csv"):
        """
        Load all matching files from a directory
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory
        file_pattern : str
            Pattern to match files
            
        Returns:
        --------
        list
            List of loaded file paths
        """
        try:
            # Find all matching files
            file_paths = glob.glob(os.path.join(directory_path, file_pattern))
            
            # Load each file
            loaded_files = []
            for file_path in file_paths:
                try:
                    self.load_file(file_path)
                    loaded_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
            
            return loaded_files
        except Exception as e:
            logger.error(f"Error loading directory {directory_path}: {e}")
            raise e
    
    def create_group(self, group_name, file_paths=None):
        """
        Create a group of FRAP files for comparison
        
        Parameters:
        -----------
        group_name : str
            Name for the group
        file_paths : list
            List of file paths to include in the group
            
        Returns:
        --------
        dict
            Dictionary containing the group data
        """
        if group_name in self.groups:
            logger.warning(f"Group {group_name} already exists, overwriting")
        
        # Initialize group
        group = {
            'name': group_name,
            'files': [],
            'features_df': None,
            'clusters': None,
            'stats': None
        }
        
        # Add files to group
        if file_paths:
            for file_path in file_paths:
                # Load file if not already loaded
                if file_path not in self.files:
                    try:
                        self.load_file(file_path)
                    except Exception as e:
                        logger.error(f"Error loading {file_path} for group {group_name}: {e}")
                        continue
                
                # Add to group
                if file_path in self.files:
                    group['files'].append(file_path)
        
        # Store group
        self.groups[group_name] = group
        self.current_group = group_name
        
        # Update group analysis
        self.update_group_analysis(group_name)
        
        return group
    
    def add_file_to_group(self, group_name, file_path):
        """
        Add a file to an existing group
        
        Parameters:
        -----------
        group_name : str
            Name of the group
        file_path : str
            Path to the file to add
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if group_name not in self.groups:
            logger.error(f"Group {group_name} does not exist")
            return False
        
        # Load file if not already loaded
        if file_path not in self.files:
            try:
                self.load_file(file_path)
            except Exception as e:
                logger.error(f"Error loading {file_path} for group {group_name}: {e}")
                return False
        
        # Add to group if not already in group
        if file_path not in self.groups[group_name]['files']:
            self.groups[group_name]['files'].append(file_path)
            
            # Update group analysis
            self.update_group_analysis(group_name)
            
        return True
    
    def remove_file_from_group(self, group_name, file_path):
        """
        Remove a file from a group
        
        Parameters:
        -----------
        group_name : str
            Name of the group
        file_path : str
            Path to the file to remove
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if group_name not in self.groups:
            logger.error(f"Group {group_name} does not exist")
            return False
        
        if file_path in self.groups[group_name]['files']:
            self.groups[group_name]['files'].remove(file_path)
            
            # Update group analysis
            self.update_group_analysis(group_name)
            
        return True
    
    def update_group_analysis(self, group_name):
        """
        Update analysis for a group (features, clustering, statistics)
        
        Parameters:
        -----------
        group_name : str
            Name of the group
            
        Returns:
        --------
        dict
            Updated group information
        """
        if group_name not in self.groups:
            logger.error(f"Group {group_name} does not exist")
            return None
        
        group = self.groups[group_name]
        
        # Create DataFrame of features from all files in the group
        features_list = []
        for file_path in group['files']:
            if file_path in self.files and self.files[file_path]['features']:
                file_features = self.files[file_path]['features'].copy()
                file_features['file_path'] = file_path
                file_features['file_name'] = self.files[file_path]['name']
                features_list.append(file_features)
        
        if features_list:
            group['features_df'] = pd.DataFrame(features_list)
            
            # Calculate basic statistics
            numerical_columns = group['features_df'].select_dtypes(include=[np.number]).columns
            group['stats'] = {
                'mean': group['features_df'][numerical_columns].mean().to_dict(),
                'std': group['features_df'][numerical_columns].std().to_dict(),
                'median': group['features_df'][numerical_columns].median().to_dict(),
                'min': group['features_df'][numerical_columns].min().to_dict(),
                'max': group['features_df'][numerical_columns].max().to_dict(),
                'count': len(group['features_df'])
            }
            
            # Only attempt clustering if we have enough samples
            if len(group['features_df']) >= 3:
                try:
                    n_clusters = min(len(group['features_df']) // 2, 5)  # Heuristic for number of clusters
                    labels, model, silhouette = FRAPAnalysisCore.perform_clustering(
                        group['features_df'], 
                        n_clusters=n_clusters, 
                        method='kmeans'
                    )
                    
                    if labels is not None:
                        group['features_df']['cluster'] = labels
                        group['clusters'] = {
                            'labels': labels,
                            'silhouette': silhouette,
                            'n_clusters': n_clusters
                        }
                except Exception as e:
                    logger.error(f"Error performing clustering for group {group_name}: {e}")
        else:
            group['features_df'] = None
            group['stats'] = None
            group['clusters'] = None
        
        # Update the group
        self.groups[group_name] = group
        
        return group
    
    def compare_groups(self, group_names, feature='half_time'):
        """
        Compare statistics between different groups
        
        Parameters:
        -----------
        group_names : list
            List of group names to compare
        feature : str
            Feature to compare between groups
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with comparison statistics
        """
        if not all(group_name in self.groups for group_name in group_names):
            missing = [group_name for group_name in group_names if group_name not in self.groups]
            logger.error(f"Groups {missing} do not exist")
            return None
        
        comparison = {}
        
        for group_name in group_names:
            group = self.groups[group_name]
            if group['stats'] and feature in group['stats']['mean']:
                comparison[group_name] = {
                    'mean': group['stats']['mean'][feature],
                    'std': group['stats']['std'][feature],
                    'median': group['stats']['median'][feature],
                    'min': group['stats']['min'][feature],
                    'max': group['stats']['max'][feature],
                    'count': group['stats']['count']
                }
        
        if comparison:
            return pd.DataFrame(comparison).T
        else:
            return None
