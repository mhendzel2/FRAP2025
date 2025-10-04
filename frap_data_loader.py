"""
Data Loading Utilities for Streamlit UI
Handles file uploads, format detection, and session state management
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

from frap_data_model import DataIO, validate_roi_traces, validate_cell_features

logger = logging.getLogger(__name__)


def render_data_loader():
    """Render data loading interface in Streamlit"""
    st.markdown("## 📂 Load FRAP Analysis Data")
    
    st.markdown("""
    Upload your analysis results or select from example datasets.
    
    **Required files:**
    - `roi_traces.parquet` (or .csv) - Time series data for each ROI
    - `cell_features.parquet` (or .csv) - Fitted parameters and QC metrics
    """)
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["📁 Upload Files", "📊 Example Data", "🔄 Recent"])
    
    with tab1:
        render_file_upload()
    
    with tab2:
        render_example_data()
    
    with tab3:
        render_recent_data()


def render_file_upload():
    """Render file upload interface"""
    st.markdown("### Upload Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        traces_file = st.file_uploader(
            "ROI Traces",
            type=['parquet', 'csv'],
            key='upload_traces',
            help="Upload roi_traces.parquet or roi_traces.csv"
        )
    
    with col2:
        features_file = st.file_uploader(
            "Cell Features",
            type=['parquet', 'csv'],
            key='upload_features',
            help="Upload cell_features.parquet or cell_features.csv"
        )
    
    if traces_file and features_file:
        if st.button("🔄 Load Data", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                success = load_from_uploads(traces_file, features_file)
                
                if success:
                    st.success("✓ Data loaded successfully!")
                    
                    # Show preview
                    st.markdown("#### Preview")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**ROI Traces**")
                        st.dataframe(st.session_state.roi_traces.head(), height=200)
                    
                    with col2:
                        st.markdown("**Cell Features**")
                        st.dataframe(st.session_state.cell_features.head(), height=200)
                    
                    # Add to recent
                    add_to_recent(traces_file.name, features_file.name)
                    
                    st.balloons()


def render_example_data():
    """Render example data selector"""
    st.markdown("### Load Example Datasets")
    
    examples = get_example_datasets()
    
    if not examples:
        st.info("No example datasets found. Run `python quick_start_singlecell.py` to generate examples.")
        return
    
    selected_example = st.selectbox(
        "Select example",
        options=list(examples.keys()),
        format_func=lambda x: f"{x} ({examples[x]['n_cells']} cells, {examples[x]['n_conditions']} conditions)"
    )
    
    if selected_example and st.button("📊 Load Example", type="primary", use_container_width=True):
        example_info = examples[selected_example]
        
        with st.spinner(f"Loading {selected_example}..."):
            success = load_from_directory(example_info['path'])
            
            if success:
                st.success(f"✓ Loaded {selected_example}")
                
                # Show info
                st.markdown("#### Dataset Info")
                st.json({
                    'name': selected_example,
                    'cells': example_info['n_cells'],
                    'conditions': example_info['n_conditions'],
                    'experiments': example_info['n_experiments'],
                    'path': str(example_info['path'])
                })


def render_recent_data():
    """Render recent datasets"""
    st.markdown("### Recently Loaded")
    
    if 'recent_datasets' not in st.session_state:
        st.session_state.recent_datasets = []
    
    if not st.session_state.recent_datasets:
        st.info("No recent datasets. Load data from Upload or Examples tab.")
        return
    
    for i, dataset in enumerate(st.session_state.recent_datasets[:5]):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{dataset['name']}**")
                st.caption(f"Loaded: {dataset['timestamp']}")
            
            with col2:
                st.metric("Cells", dataset['n_cells'])
            
            with col3:
                if st.button("🔄 Reload", key=f"reload_{i}"):
                    if 'path' in dataset:
                        load_from_directory(Path(dataset['path']))
                        st.rerun()
            
            st.divider()


def load_from_uploads(traces_file, features_file) -> bool:
    """Load data from uploaded files"""
    try:
        # Detect format
        traces_format = Path(traces_file.name).suffix.lower()
        features_format = Path(features_file.name).suffix.lower()
        
        # Load traces
        if traces_format == '.parquet':
            roi_traces = pd.read_parquet(traces_file)
        elif traces_format == '.csv':
            roi_traces = pd.read_csv(traces_file)
        else:
            st.error(f"Unsupported format: {traces_format}")
            return False
        
        # Load features
        if features_format == '.parquet':
            cell_features = pd.read_parquet(features_file)
        elif features_format == '.csv':
            cell_features = pd.read_csv(features_file)
        else:
            st.error(f"Unsupported format: {features_format}")
            return False
        
        # Validate
        validate_and_load(roi_traces, cell_features)
        
        return True
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.exception("Failed to load uploaded files")
        return False


def load_from_directory(directory: Path) -> bool:
    """Load data from directory with DataIO"""
    try:
        directory = Path(directory)
        
        if not directory.exists():
            st.error(f"Directory not found: {directory}")
            return False
        
        io = DataIO()
        roi_traces, cell_features = io.load_tables(directory)
        
        validate_and_load(roi_traces, cell_features)
        
        return True
        
    except Exception as e:
        st.error(f"Error loading from directory: {e}")
        logger.exception(f"Failed to load from {directory}")
        return False


def validate_and_load(roi_traces: pd.DataFrame, cell_features: pd.DataFrame):
    """Validate and load data into session state"""
    # Validate schema
    traces_valid, traces_msg = validate_roi_traces(roi_traces)
    if not traces_valid:
        st.error(f"Invalid ROI traces: {traces_msg}")
        raise ValueError(traces_msg)
    
    features_valid, features_msg = validate_cell_features(cell_features)
    if not features_valid:
        st.error(f"Invalid cell features: {features_msg}")
        raise ValueError(features_msg)
    
    # Check consistency
    trace_cells = set(roi_traces['cell_id'].unique())
    feature_cells = set(cell_features['cell_id'].unique())
    
    if not trace_cells.issubset(feature_cells):
        missing = trace_cells - feature_cells
        st.warning(f"Warning: {len(missing)} cells in traces but not in features")
    
    # Load into session state
    st.session_state.roi_traces = roi_traces
    st.session_state.cell_features = cell_features
    
    logger.info(f"Loaded {len(roi_traces)} trace rows, {len(cell_features)} cells")


def get_example_datasets() -> dict:
    """Find available example datasets"""
    examples = {}
    
    # Check common locations
    search_paths = [
        Path('./output'),
        Path('./data'),
        Path('./examples'),
        Path('../output'),
    ]
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        # Look for subdirectories with parquet files
        for subdir in base_path.iterdir():
            if not subdir.is_dir():
                continue
            
            traces_file = subdir / 'roi_traces.parquet'
            features_file = subdir / 'cell_features.parquet'
            
            if traces_file.exists() and features_file.exists():
                try:
                    # Quick check
                    features = pd.read_parquet(features_file)
                    
                    examples[subdir.name] = {
                        'path': subdir,
                        'n_cells': len(features),
                        'n_conditions': features['condition'].nunique() if 'condition' in features.columns else 1,
                        'n_experiments': features['exp_id'].nunique()
                    }
                except:
                    pass
    
    return examples


def add_to_recent(traces_name: str, features_name: str):
    """Add dataset to recent list"""
    if 'recent_datasets' not in st.session_state:
        st.session_state.recent_datasets = []
    
    dataset_info = {
        'name': f"{Path(traces_name).stem}",
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_cells': len(st.session_state.cell_features)
    }
    
    # Add to front, keep max 10
    st.session_state.recent_datasets.insert(0, dataset_info)
    st.session_state.recent_datasets = st.session_state.recent_datasets[:10]


def export_current_cohort(format: str = 'csv') -> Tuple[bytes, str]:
    """Export current cohort for download"""
    from io import BytesIO
    
    # Get filtered data
    df = st.session_state.cell_features.copy()
    
    # Apply filters
    filters = st.session_state.active_filters
    
    if 'condition' in filters and filters['condition']:
        df = df[df['condition'].isin(filters['condition'])]
    
    if 'exp_id' in filters and filters['exp_id']:
        df = df[df['exp_id'].isin(filters['exp_id'])]
    
    if 'clusters' in filters and filters['clusters']:
        df = df[df['cluster'].isin(filters['clusters'])]
    
    if 'outliers' in filters and not filters['outliers']:
        df = df[~df['outlier']]
    
    if 'qc_pass' in filters and filters['qc_pass']:
        df = df[df['bleach_qc'] == True]
    
    # Export
    if format == 'csv':
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue(), 'cohort_export.csv'
    
    elif format == 'parquet':
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue(), 'cohort_export.parquet'
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_traces(cell_ids: list, format: str = 'csv') -> Tuple[bytes, str]:
    """Export traces for specific cells"""
    from io import BytesIO
    
    traces = st.session_state.roi_traces[
        st.session_state.roi_traces['cell_id'].isin(cell_ids)
    ]
    
    if format == 'csv':
        buffer = BytesIO()
        traces.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue(), 'traces_export.csv'
    
    elif format == 'parquet':
        buffer = BytesIO()
        traces.to_parquet(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue(), 'traces_export.parquet'
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def check_data_quality() -> dict:
    """Run quality checks on loaded data"""
    issues = {
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    if st.session_state.cell_features.empty:
        issues['errors'].append("No data loaded")
        return issues
    
    df = st.session_state.cell_features
    traces = st.session_state.roi_traces
    
    # Check for required columns
    required_cols = ['cell_id', 'exp_id', 'mobile_frac', 'k', 't_half', 'r2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        issues['errors'].append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for NaN values
    if df[required_cols].isna().any().any():
        nan_counts = df[required_cols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        issues['warnings'].append(f"NaN values found: {nan_cols.to_dict()}")
    
    # Check QC pass rate
    if 'bleach_qc' in df.columns:
        pass_rate = df['bleach_qc'].mean()
        if pass_rate < 0.5:
            issues['warnings'].append(f"Low QC pass rate: {pass_rate*100:.1f}%")
        else:
            issues['info'].append(f"QC pass rate: {pass_rate*100:.1f}%")
    
    # Check R² distribution
    if 'r2' in df.columns:
        median_r2 = df['r2'].median()
        if median_r2 < 0.7:
            issues['warnings'].append(f"Low median R²: {median_r2:.3f}")
        else:
            issues['info'].append(f"Median R²: {median_r2:.3f}")
    
    # Check drift
    if 'drift_px' in df.columns:
        high_drift = (df['drift_px'] > 10).sum()
        if high_drift > len(df) * 0.2:
            issues['warnings'].append(f"High drift in {high_drift}/{len(df)} cells")
        else:
            issues['info'].append(f"High drift in {high_drift}/{len(df)} cells")
    
    # Check trace coverage
    cells_with_traces = traces['cell_id'].nunique()
    cells_in_features = len(df)
    
    if cells_with_traces < cells_in_features:
        issues['warnings'].append(
            f"Only {cells_with_traces}/{cells_in_features} cells have trace data"
        )
    
    return issues


if __name__ == '__main__':
    # Test
    print("Finding example datasets...")
    examples = get_example_datasets()
    
    for name, info in examples.items():
        print(f"\n{name}:")
        print(f"  Path: {info['path']}")
        print(f"  Cells: {info['n_cells']}")
        print(f"  Conditions: {info['n_conditions']}")
        print(f"  Experiments: {info['n_experiments']}")
