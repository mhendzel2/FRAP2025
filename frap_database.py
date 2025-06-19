"""
FRAP Database Module
Manage database operations for storing FRAP analysis results
"""
import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, 
    DateTime, ForeignKey, Text, Boolean, inspect,
    JSON, ARRAY
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
Base = declarative_base()

class Experiment(Base):
    """Experiment table for grouping related FRAP measurements"""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationship
    frap_files = relationship("FRAPFile", back_populates="experiment", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Experiment(name='{self.name}')>"

class FRAPFile(Base):
    """FRAPFile table for storing individual FRAP measurement files"""
    __tablename__ = 'frap_files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    uploaded_at = Column(DateTime, default=datetime.now)
    
    # Store raw data as JSON
    raw_data_json = Column(Text)
    processed_data_json = Column(Text)
    
    # Analysis parameters
    bleach_spot_radius = Column(Float)
    pixel_size = Column(Float)
    reference_d_value = Column(Float, default=25.0)  # GFP reference in μm²/s
    
    # Relationship
    experiment = relationship("Experiment", back_populates="frap_files")
    fits = relationship("FRAPFit", back_populates="frap_file", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<FRAPFile(filename='{self.filename}')>"
    
    def set_raw_data(self, df):
        """Store raw data DataFrame as JSON"""
        self.raw_data_json = df.to_json()
        
    def get_raw_data(self):
        """Retrieve raw data DataFrame from JSON"""
        if self.raw_data_json:
            return pd.read_json(self.raw_data_json)
        return None
    
    def set_processed_data(self, df):
        """Store processed data DataFrame as JSON"""
        self.processed_data_json = df.to_json()
        
    def get_processed_data(self):
        """Retrieve processed data DataFrame from JSON"""
        if self.processed_data_json:
            return pd.read_json(self.processed_data_json)
        return None

class FRAPFit(Base):
    """FRAPFit table for storing model fitting results"""
    __tablename__ = 'frap_fits'
    
    id = Column(Integer, primary_key=True)
    frap_file_id = Column(Integer, ForeignKey('frap_files.id'))
    model_type = Column(String(50))  # 'single', 'double', 'triple'
    
    # Fit parameters
    parameters_json = Column(Text)  # Store parameters as JSON
    
    # Fit statistics
    r_squared = Column(Float)
    adjusted_r_squared = Column(Float)
    aic = Column(Float)
    bic = Column(Float)
    
    # Is this the best fit for this file?
    is_best_fit = Column(Boolean, default=False)
    
    # Features for clustering
    mobile_fraction = Column(Float)
    half_time = Column(Float)
    
    # Diffusion-related parameters
    diffusion_coefficient = Column(Float)  # For single component or main component
    radius_of_gyration = Column(Float)
    molecular_weight_estimate = Column(Float)
    
    # Multiple component additional info
    component_count = Column(Integer)
    rate_constants = Column(ARRAY(Float))
    amplitudes = Column(ARRAY(Float))
    proportions = Column(ARRAY(Float))
    diffusion_coefficients = Column(ARRAY(Float))
    radii_of_gyration = Column(ARRAY(Float))
    molecular_weight_estimates = Column(ARRAY(Float))
    
    # Relationship
    frap_file = relationship("FRAPFile", back_populates="fits")
    
    def __repr__(self):
        return f"<FRAPFit(model='{self.model_type}', r²={self.r_squared:.4f})>"
    
    def set_parameters(self, params_dict):
        """Store parameters dictionary as JSON"""
        self.parameters_json = json.dumps(params_dict)
        
    def get_parameters(self):
        """Retrieve parameters dictionary from JSON"""
        if self.parameters_json:
            return json.loads(self.parameters_json)
        return {}

class FRAPGroup(Base):
    """FRAPGroup table for storing analysis groups"""
    __tablename__ = 'frap_groups'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Clustering parameters
    n_clusters = Column(Integer)
    clustering_method = Column(String(50))
    silhouette_score = Column(Float)
    
    # Group statistics
    statistics_json = Column(Text)  # Store statistics as JSON
    
    # Relationship
    group_files = relationship("GroupFile", back_populates="group", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<FRAPGroup(name='{self.name}')>"
    
    def set_statistics(self, stats_dict):
        """Store statistics dictionary as JSON"""
        self.statistics_json = json.dumps(stats_dict)
        
    def get_statistics(self):
        """Retrieve statistics dictionary from JSON"""
        if self.statistics_json:
            return json.loads(self.statistics_json)
        return {}

class GroupFile(Base):
    """GroupFile table for mapping files to groups"""
    __tablename__ = 'group_files'
    
    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey('frap_groups.id'))
    frap_file_id = Column(Integer, ForeignKey('frap_files.id'))
    cluster_label = Column(Integer)
    
    # Relationships
    group = relationship("FRAPGroup", back_populates="group_files")
    
    def __repr__(self):
        return f"<GroupFile(group_id={self.group_id}, file_id={self.frap_file_id})>"

class FRAPDatabase:
    """Database manager for FRAP analysis"""
    
    def __init__(self, database_url=None):
        """Initialize database connection"""
        if database_url is None:
            database_url = os.environ.get('DATABASE_URL')
            
        if not database_url:
            raise ValueError("Database URL is required. Set the DATABASE_URL environment variable.")
            
        # Create engine and session
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Check if tables exist
        self.init_db()
            
    def init_db(self):
        """Initialize database tables if they don't exist"""
        inspector = inspect(self.engine)
        if not inspector.has_table('experiments'):
            try:
                Base.metadata.create_all(self.engine)
                logger.info("Database tables created successfully")
            except Exception as e:
                logger.error(f"Error creating database tables: {e}")
                raise
                
    # Experiment methods
    def create_experiment(self, name, description=None):
        """Create a new experiment"""
        session = self.Session()
        try:
            experiment = Experiment(name=name, description=description)
            session.add(experiment)
            session.commit()
            experiment_id = experiment.id
            logger.info(f"Created experiment: {name} (ID: {experiment_id})")
            return experiment_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating experiment: {e}")
            raise
        finally:
            session.close()
            
    def get_experiment(self, experiment_id):
        """Get experiment by ID"""
        session = self.Session()
        try:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            return experiment
        except Exception as e:
            logger.error(f"Error retrieving experiment {experiment_id}: {e}")
            raise
        finally:
            session.close()
            
    def get_experiments(self):
        """Get all experiments"""
        session = self.Session()
        try:
            experiments = session.query(Experiment).all()
            return experiments
        except Exception as e:
            logger.error(f"Error retrieving experiments: {e}")
            raise
        finally:
            session.close()
            
    def delete_experiment(self, experiment_id):
        """Delete experiment and all associated files"""
        session = self.Session()
        try:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                session.delete(experiment)
                session.commit()
                logger.info(f"Deleted experiment: {experiment.name} (ID: {experiment_id})")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting experiment {experiment_id}: {e}")
            raise
        finally:
            session.close()
            
    # FRAP file methods
    def add_frap_file(self, filename, experiment_id, raw_data, processed_data, 
                      bleach_spot_radius=1.0, pixel_size=1.0, reference_d_value=25.0):
        """Add FRAP file to database"""
        session = self.Session()
        try:
            frap_file = FRAPFile(
                filename=filename,
                experiment_id=experiment_id,
                bleach_spot_radius=bleach_spot_radius,
                pixel_size=pixel_size,
                reference_d_value=reference_d_value
            )
            
            # Store DataFrames as JSON
            frap_file.set_raw_data(raw_data)
            frap_file.set_processed_data(processed_data)
            
            session.add(frap_file)
            session.commit()
            file_id = frap_file.id
            logger.info(f"Added FRAP file: {filename} (ID: {file_id})")
            return file_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding FRAP file: {e}")
            raise
        finally:
            session.close()
            
    def get_frap_file(self, file_id):
        """Get FRAP file by ID"""
        session = self.Session()
        try:
            frap_file = session.query(FRAPFile).filter(FRAPFile.id == file_id).first()
            return frap_file
        except Exception as e:
            logger.error(f"Error retrieving FRAP file {file_id}: {e}")
            raise
        finally:
            session.close()
            
    def get_frap_files_by_experiment(self, experiment_id):
        """Get all FRAP files in an experiment"""
        session = self.Session()
        try:
            files = session.query(FRAPFile).filter(FRAPFile.experiment_id == experiment_id).all()
            return files
        except Exception as e:
            logger.error(f"Error retrieving FRAP files for experiment {experiment_id}: {e}")
            raise
        finally:
            session.close()
            
    def delete_frap_file(self, file_id):
        """Delete FRAP file and all associated fits"""
        session = self.Session()
        try:
            frap_file = session.query(FRAPFile).filter(FRAPFile.id == file_id).first()
            if frap_file:
                session.delete(frap_file)
                session.commit()
                logger.info(f"Deleted FRAP file: {frap_file.filename} (ID: {file_id})")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting FRAP file {file_id}: {e}")
            raise
        finally:
            session.close()
            
    # FRAP fit methods
    def add_frap_fit(self, frap_file_id, model_type, parameters, stats, features, is_best_fit=False):
        """Add fit result to database"""
        session = self.Session()
        try:
            # If this is the best fit, set all other fits for this file to not best
            if is_best_fit:
                existing_best = session.query(FRAPFit).filter(
                    FRAPFit.frap_file_id == frap_file_id, 
                    FRAPFit.is_best_fit == True
                ).all()
                
                for fit in existing_best:
                    fit.is_best_fit = False
            
            # Create new fit record
            frap_fit = FRAPFit(
                frap_file_id=frap_file_id,
                model_type=model_type,
                is_best_fit=is_best_fit,
                r_squared=stats.get('r2'),
                adjusted_r_squared=stats.get('adj_r2'),
                aic=stats.get('aic'),
                bic=stats.get('bic'),
                mobile_fraction=features.get('mobile_fraction'),
                half_time=features.get('half_time'),
                component_count=1 if model_type == 'single' else (2 if model_type == 'double' else 3)
            )
            
            # Set parameters
            frap_fit.set_parameters(parameters)
            
            # Set diffusion-related fields
            if model_type == 'single':
                frap_fit.diffusion_coefficient = features.get('diffusion_coefficient')
                frap_fit.radius_of_gyration = features.get('radius_of_gyration')
                frap_fit.molecular_weight_estimate = features.get('molecular_weight_estimate')
                
                # Set arrays for consistency
                frap_fit.rate_constants = [features.get('rate_constant')]
                frap_fit.amplitudes = [features.get('amplitude')]
                frap_fit.proportions = [1.0]
                frap_fit.diffusion_coefficients = [features.get('diffusion_coefficient')]
                frap_fit.radii_of_gyration = [features.get('radius_of_gyration')]
                frap_fit.molecular_weight_estimates = [features.get('molecular_weight_estimate')]
            else:
                # For multiple components, store the primary (fastest) component values
                component_count = 2 if model_type == 'double' else 3
                
                # Extract arrays for all components
                rate_constants = []
                amplitudes = []
                proportions = []
                diffusion_coeffs = []
                radii = []
                mw_estimates = []
                
                for i in range(1, component_count + 1):
                    rate_constants.append(features.get(f'rate_constant_{i}', 0))
                    amplitudes.append(features.get(f'amplitude_{i}', 0))
                    proportions.append(features.get(f'proportion_{i}', 0))
                    diffusion_coeffs.append(features.get(f'diffusion_coefficient_{i}', 0))
                    radii.append(features.get(f'radius_of_gyration_{i}', 0))
                    mw_estimates.append(features.get(f'molecular_weight_estimate_{i}', 0))
                
                # Store arrays
                frap_fit.rate_constants = rate_constants
                frap_fit.amplitudes = amplitudes
                frap_fit.proportions = proportions
                frap_fit.diffusion_coefficients = diffusion_coeffs
                frap_fit.radii_of_gyration = radii
                frap_fit.molecular_weight_estimates = mw_estimates
                
                # Store main component values (fastest, usually component 1)
                frap_fit.diffusion_coefficient = diffusion_coeffs[0] if diffusion_coeffs else None
                frap_fit.radius_of_gyration = radii[0] if radii else None
                frap_fit.molecular_weight_estimate = mw_estimates[0] if mw_estimates else None
            
            session.add(frap_fit)
            session.commit()
            fit_id = frap_fit.id
            logger.info(f"Added FRAP fit: {model_type} for file ID {frap_file_id} (Fit ID: {fit_id})")
            return fit_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding FRAP fit: {e}")
            raise
        finally:
            session.close()
            
    def get_frap_fits(self, frap_file_id):
        """Get all fits for a FRAP file"""
        session = self.Session()
        try:
            fits = session.query(FRAPFit).filter(FRAPFit.frap_file_id == frap_file_id).all()
            return fits
        except Exception as e:
            logger.error(f"Error retrieving FRAP fits for file {frap_file_id}: {e}")
            raise
        finally:
            session.close()
            
    def get_best_fit(self, frap_file_id):
        """Get best fit for a FRAP file"""
        session = self.Session()
        try:
            best_fit = session.query(FRAPFit).filter(
                FRAPFit.frap_file_id == frap_file_id,
                FRAPFit.is_best_fit == True
            ).first()
            return best_fit
        except Exception as e:
            logger.error(f"Error retrieving best fit for file {frap_file_id}: {e}")
            raise
        finally:
            session.close()
            
    # Group methods
    def create_group(self, name, description=None):
        """Create a new group"""
        session = self.Session()
        try:
            group = FRAPGroup(name=name, description=description)
            session.add(group)
            session.commit()
            group_id = group.id
            logger.info(f"Created group: {name} (ID: {group_id})")
            return group_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating group: {e}")
            raise
        finally:
            session.close()
            
    def get_group(self, group_id):
        """Get group by ID"""
        session = self.Session()
        try:
            group = session.query(FRAPGroup).filter(FRAPGroup.id == group_id).first()
            return group
        except Exception as e:
            logger.error(f"Error retrieving group {group_id}: {e}")
            raise
        finally:
            session.close()
            
    def get_groups(self):
        """Get all groups"""
        session = self.Session()
        try:
            groups = session.query(FRAPGroup).all()
            return groups
        except Exception as e:
            logger.error(f"Error retrieving groups: {e}")
            raise
        finally:
            session.close()
            
    def add_file_to_group(self, group_id, frap_file_id, cluster_label=None):
        """Add FRAP file to group"""
        session = self.Session()
        try:
            # Check if file is already in group
            existing = session.query(GroupFile).filter(
                GroupFile.group_id == group_id,
                GroupFile.frap_file_id == frap_file_id
            ).first()
            
            if existing:
                # Update cluster label if provided
                if cluster_label is not None:
                    existing.cluster_label = cluster_label
                    session.commit()
                    logger.info(f"Updated cluster label for file {frap_file_id} in group {group_id}")
                return existing.id
                
            # Add new association
            group_file = GroupFile(
                group_id=group_id,
                frap_file_id=frap_file_id,
                cluster_label=cluster_label
            )
            session.add(group_file)
            session.commit()
            association_id = group_file.id
            logger.info(f"Added file {frap_file_id} to group {group_id}")
            return association_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding file to group: {e}")
            raise
        finally:
            session.close()
            
    def remove_file_from_group(self, group_id, frap_file_id):
        """Remove FRAP file from group"""
        session = self.Session()
        try:
            association = session.query(GroupFile).filter(
                GroupFile.group_id == group_id,
                GroupFile.frap_file_id == frap_file_id
            ).first()
            
            if association:
                session.delete(association)
                session.commit()
                logger.info(f"Removed file {frap_file_id} from group {group_id}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing file from group: {e}")
            raise
        finally:
            session.close()
            
    def get_files_in_group(self, group_id):
        """Get all files in a group"""
        session = self.Session()
        try:
            associations = session.query(GroupFile).filter(GroupFile.group_id == group_id).all()
            file_ids = [assoc.frap_file_id for assoc in associations]
            
            # Get actual file objects
            files = []
            for file_id in file_ids:
                file = session.query(FRAPFile).filter(FRAPFile.id == file_id).first()
                if file:
                    files.append(file)
                    
            return files
        except Exception as e:
            logger.error(f"Error retrieving files in group {group_id}: {e}")
            raise
        finally:
            session.close()
            
    def update_group_clustering(self, group_id, n_clusters, method, silhouette_score):
        """Update group clustering information"""
        session = self.Session()
        try:
            group = session.query(FRAPGroup).filter(FRAPGroup.id == group_id).first()
            if group:
                group.n_clusters = n_clusters
                group.clustering_method = method
                group.silhouette_score = silhouette_score
                session.commit()
                logger.info(f"Updated clustering for group {group_id}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating group clustering: {e}")
            raise
        finally:
            session.close()
            
    def update_group_statistics(self, group_id, statistics):
        """Update group statistics"""
        session = self.Session()
        try:
            group = session.query(FRAPGroup).filter(FRAPGroup.id == group_id).first()
            if group:
                group.set_statistics(statistics)
                session.commit()
                logger.info(f"Updated statistics for group {group_id}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating group statistics: {e}")
            raise
        finally:
            session.close()