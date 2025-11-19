"""
UI Components for the FRAP Analysis App
This module contains functions that create the Streamlit UI panels for
motion stabilization, calibration, and statistical analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import linregress, ttest_ind, f_oneway
from frap_image_analysis import FRAPImageAnalyzer
from frap_core_corrected import FRAPAnalysisCore

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg procedure for controlling the False Discovery Rate.
    """
    p_values = np.asfarray(p_values)
    by_descend = p_values.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p_values)) / np.arange(len(p_values), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p_values[by_descend]))
    return q[by_orig] < alpha, q[by_orig]

def create_sparkline(x_coords, y_coords):
    """Generates a tiny sparkline for centroid displacement."""
    if not isinstance(x_coords, (list, np.ndarray, pd.Series)) or not isinstance(y_coords, (list, np.ndarray, pd.Series)):
        return ""

    x = np.array(x_coords)
    y = np.array(y_coords)

    if len(x) < 2 or len(y) < 2 or pd.isnull(x).all() or pd.isnull(y).all():
        return ""

    # Calculate displacement from the initial point
    displacement = np.sqrt((x - x[0])**2 + (y - y[0])**2)

    fig = go.Figure(go.Scatter(
        x=np.arange(len(displacement)), y=displacement,
        mode='lines', line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        width=100, height=30,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig.to_image(format="svg+xml", engine="kaleido")


def create_motion_stabilization_panel(analyzer: FRAPImageAnalyzer):
    """Creates the UI panel for motion stabilization settings."""
    st.header("Motion Stabilization")

    if analyzer.image_stack is None:
        st.warning("Please load an image stack in the 'Image Analysis' tab to use this feature.")
        return

    st.subheader("Stabilization Settings")

    st.session_state.app_config['motion']['global_drift'] = st.checkbox(
        "Correct for global drift",
        value=st.session_state.app_config['motion']['global_drift'],
        help="Applies a global translation registration to all frames before local tracking."
    )
    st.session_state.app_config['motion']['optical_flow'] = st.checkbox(
        "Use local optical flow",
        value=st.session_state.app_config['motion']['optical_flow'],
        help="Uses Lucas-Kanade optical flow to predict spot movement between frames, improving tracking robustness."
    )

    if st.button("Preview Stabilization", type="primary"):
        with st.spinner("Running motion stabilization preview..."):
            try:
                if analyzer.bleach_coordinates is None or not analyzer.rois:
                    st.error("Bleach spot not defined. Please detect it in the 'Image Analysis' tab first.")
                    return

                bleach_roi_data = next((roi for roi in analyzer.rois.values() if roi['type'] == 'bleach_spot'), None)
                if bleach_roi_data is None:
                    st.error("Bleach spot ROI not found.")
                    return

                results = FRAPAnalysisCore.motion_compensate_stack(
                    stack=analyzer.image_stack,
                    init_center=analyzer.bleach_coordinates,
                    radius=bleach_roi_data.get('radius', 10),
                    pixel_size_um=analyzer.pixel_size,
                    use_optical_flow=st.session_state.app_config['motion']['optical_flow'],
                    do_global=st.session_state.app_config['motion']['global_drift'],
                    kalman=True
                )
                st.session_state.stabilization_preview_results = results
                st.success("Motion stabilization preview complete.")

            except Exception as e:
                st.error(f"Motion stabilization preview failed: {e}")

    if 'stabilization_preview_results' in st.session_state and st.session_state.stabilization_preview_results:
        results = st.session_state.stabilization_preview_results
        st.subheader("Stabilization Preview Results")

        warnings = results.get('warnings', [])
        if warnings:
            for warning in warnings:
                st.warning(warning)

        col1, col2 = st.columns(2)
        with col1:
            drift_um = results.get('drift_um')
            if drift_um is not None:
                st.metric("Total Drift", f"{drift_um:.2f} µm")
            else:
                st.metric("Total Drift", "N/A (pixel size not set)")

        with col2:
            roi_trace = results.get('roi_trace', [])
            if roi_trace:
                x_coords = [d['centroid']['x'] for d in roi_trace]
                y_coords = [d['centroid']['y'] for d in roi_trace]
                sparkline_svg = create_sparkline(x_coords, y_coords)
                st.markdown("**Centroid Displacement**")
                st.image(sparkline_svg)

        with st.expander("Show Drift Details"):
            if roi_trace:
                df = pd.DataFrame(roi_trace)
                df_display = df[['frame', 'displacement_px']].copy()
                df_display['centroid_x'] = [d['centroid']['x'] for d in df['centroid']]
                df_display['centroid_y'] = [d['y'] for d in df['centroid']]
                st.dataframe(df_display)

                fig = px.line(df_display, x='frame', y='displacement_px', title='Per-frame Displacement')
                st.plotly_chart(fig, use_container_width=True)


def create_calibration_panel():
    """Creates the UI panel for calibration of molecular weight standards."""
    st.header("Molecular Weight Calibration")
    st.markdown("Calibrate the apparent molecular weight estimation using known standards.")

    # Editable table for standards
    st.subheader("Enter Calibration Standards")

    if 'calibration_standards' not in st.session_state:
        st.session_state.calibration_standards = pd.DataFrame([
            {"Molecule": "GFP", "MW (kDa)": 27, "D (μm²/s)": 25.0},
            {"Molecule": "mCherry", "MW (kDa)": 29, "D (μm²/s)": 22.0},
        ])

    edited_df = st.data_editor(
        st.session_state.calibration_standards,
        num_rows_editable=True,
        key="calibration_editor"
    )
    st.session_state.calibration_standards = edited_df

    if st.button("Run Calibration", type="primary"):
        df = st.session_state.calibration_standards
        if len(df) < 2:
            st.error("Please provide at least 2 standards for calibration.")
            return

        try:
            # Log-transform data
            log_mw = np.log10(df["MW (kDa)"])
            log_d = np.log10(df["D (μm²/s)"])

            # Perform linear regression
            fit = linregress(log_d, log_mw)

            st.session_state.app_config['calibration']['fit_params'] = {
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r_value": fit.rvalue
            }
            st.success(f"Calibration successful! R-squared: {fit.rvalue**2:.4f}")

        except Exception as e:
            st.error(f"Calibration failed: {e}")

    if st.session_state.app_config['calibration']['fit_params']:
        st.subheader("Calibration Results")
        fit_params = st.session_state.app_config['calibration']['fit_params']

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Slope", f"{fit_params['slope']:.4f}")
        with col2:
            st.metric("R-squared", f"{fit_params['r_value']**2:.4f}")

        # Preview plot
        df = st.session_state.calibration_standards
        log_mw = np.log10(df["MW (kDa)"])
        log_d = np.log10(df["D (μm²/s)"])

        fig = px.scatter(
            x=log_d,
            y=log_mw,
            labels={"x": "log10(D)", "y": "log10(MW)"},
            title="Calibration Curve: log(MW) vs. log(D)"
        )

        # Add regression line
        x_fit = np.linspace(log_d.min(), log_d.max(), 100)
        y_fit = fit_params['slope'] * x_fit + fit_params['intercept']
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fit'))

        st.plotly_chart(fig, use_container_width=True)


def create_stats_panel(dm):
    """Creates the UI panel for statistical analysis."""
    st.header("Statistical Analysis")
    st.markdown("Perform statistical tests to compare groups.")

    if len(dm.groups) < 2:
        st.warning("You need at least 2 groups to perform statistical comparisons.")
        return

    # Group selection
    st.subheader("Select Groups for Comparison")
    all_groups = list(dm.groups.keys())
    selected_groups = st.multiselect("Groups", all_groups, default=all_groups[:2])

    if len(selected_groups) < 2:
        st.warning("Please select at least 2 groups.")
        return

    # Parameter selection
    st.subheader("Select Parameter to Test")
    # Combine data from selected groups to find common parameters
    all_params = set()
    for group_name in selected_groups:
        if group_name in dm.groups and dm.groups[group_name].get('features_df') is not None:
            all_params.update(dm.groups[group_name]['features_df'].select_dtypes(include=np.number).columns)

    if not all_params:
        st.warning("No numeric parameters found in the selected groups.")
        return

    param_to_test = st.selectbox("Parameter", sorted(list(all_params)))

    # Test selection and options
    st.subheader("Test Settings")
    test_type = st.selectbox("Test Type", ["t-test (2 groups)", "ANOVA (>2 groups)"])

    tost_threshold = st.number_input("TOST Equivalence Threshold (%)", min_value=0.1, value=20.0, step=1.0, help="Threshold for Two One-Sided Tests (TOST) for equivalence.")
    fdr_correction = st.checkbox("Apply FDR Correction (Benjamini-Hochberg)", value=False)

    if st.button("Run Statistical Analysis", type="primary"):
        results = []
        if test_type == "t-test (2 groups)" and len(selected_groups) == 2:
            group1_name, group2_name = selected_groups
            data1 = dm.groups[group1_name]['features_df'][param_to_test].dropna()
            data2 = dm.groups[group2_name]['features_df'][param_to_test].dropna()

            # t-test
            t_stat, p_value = ttest_ind(data1, data2, equal_var=False) # Welch's t-test

            # TOST
            mean1, mean2 = data1.mean(), data2.mean()
            std1, std2 = data1.std(), data2.std()
            n1, n2 = len(data1), len(data2)

            se_diff = np.sqrt(std1**2/n1 + std2**2/n2) if n1 > 0 and n2 > 0 else 0
            lower_bound = -tost_threshold/100 * mean1
            upper_bound = tost_threshold/100 * mean1

            tost_p = np.nan
            if se_diff > 0:
                t_lower = (mean2 - mean1 - lower_bound) / se_diff
                t_upper = (mean2 - mean1 - upper_bound) / se_diff

                p_lower = ttest_ind(data1, data2, equal_var=False)[1] / 2
                p_upper = ttest_ind(data1, data2, equal_var=False)[1] / 2

                tost_p = max(p_lower, p_upper)

            results.append({
                "Comparison": f"{group1_name} vs {group2_name}",
                "Parameter": param_to_test,
                "t-statistic": t_stat,
                "p-value": p_value,
                "TOST p-value": tost_p
            })

        elif test_type == "ANOVA (>2 groups)":
            groups_data = [dm.groups[name]['features_df'][param_to_test].dropna() for name in selected_groups]
            f_stat, p_value = f_oneway(*groups_data)
            results.append({
                "Comparison": " vs ".join(selected_groups),
                "Parameter": param_to_test,
                "F-statistic": f_stat,
                "p-value": p_value
            })

        if results:
            results_df = pd.DataFrame(results)

            if fdr_correction:
                p_values = results_df['p-value']
                reject, p_adjusted = benjamini_hochberg(p_values)
                results_df['p-adjusted (FDR)'] = p_adjusted
                results_df['Significant (FDR)'] = reject

            st.subheader("Results")
            st.dataframe(results_df)
        else:
            st.warning("Could not perform the selected test with the chosen groups.")
