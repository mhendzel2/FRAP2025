#!/usr/bin/env python3
"""
Integration utilities for the FRAP Reference Database.

This module provides helper functions to integrate reference database 
comparisons into FRAP analysis workflows.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Optional, List
import plotly.graph_objects as go
from frap_reference_database import frap_reference_db

def add_reference_comparison_to_results(results: Dict, 
                                       experimental_deff: Optional[float] = None,
                                       experimental_mf: Optional[float] = None,
                                       protein_name: Optional[str] = None,
                                       compartment: Optional[str] = None) -> Dict:
    """
    Add reference database comparison to existing FRAP analysis results.
    
    Parameters:
    -----------
    results : Dict
        Existing analysis results dictionary
    experimental_deff : float, optional
        Experimental diffusion coefficient
    experimental_mf : float, optional
        Experimental mobile fraction
    protein_name : str, optional
        Name of the protein being studied
    compartment : str, optional
        Cellular compartment
    
    Returns:
    --------
    Dict
        Enhanced results with reference comparisons
    """
    
    if not any(v is not None for v in [experimental_deff, experimental_mf]):
        return results
    
    # Get reference comparison
    comparison = frap_reference_db.compare_experimental_to_reference(
        experimental_deff=experimental_deff,
        experimental_mf=experimental_mf,
        compartment=compartment
    )
    
    # Add reference data to results
    results['reference_comparison'] = comparison
    
    return results

def display_reference_comparison_widget(experimental_deff: Optional[float] = None,
                                       experimental_mf: Optional[float] = None,
                                       protein_name: Optional[str] = None,
                                       compartment: Optional[str] = None,
                                       key_suffix: str = ""):
    """
    Display a compact reference database comparison widget.
    
    This can be embedded in analysis results to provide immediate context.
    """
    
    if not any(v is not None for v in [experimental_deff, experimental_mf]):
        return
    
    with st.expander("📚 Compare with Reference Database", expanded=False):
        comparison = frap_reference_db.compare_experimental_to_reference(
            experimental_deff=experimental_deff,
            experimental_mf=experimental_mf,
            compartment=compartment
        )
        
        # Display interpretation
        if comparison['interpretation']:
            st.info(f"**Interpretation:** {comparison['interpretation']}")
        
        # Display top matches
        if comparison['closest_matches'] is not None and len(comparison['closest_matches']) > 0:
            st.markdown("**🎯 Top Reference Matches:**")
            
            top_matches = comparison['closest_matches'].head(3)
            for idx, (_, match) in enumerate(top_matches.iterrows(), 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **{idx}. {match['protein_probe']}** ({match['class_type']})
                        - Location: {match['cellular_compartment']}
                        - Finding: {match['key_finding']}
                        """)
                    
                    with col2:
                        if pd.notna(match['deff_um2_s']):
                            st.metric("D_eff", f"{match['deff_um2_s']:.2f} µm²/s")
                        if pd.notna(match['mobile_fraction_pct']):
                            st.metric("Mf", f"{match['mobile_fraction_pct']:.0f}%")
        
        # Display recommendations
        if comparison['recommendations']:
            st.markdown("**💡 Recommendations:**")
            for rec in comparison['recommendations']:
                st.markdown(f"- {rec}")

def get_reference_context_for_protein(protein_name: str) -> Optional[Dict]:
    """
    Get reference context for a specific protein if it exists in the database.
    
    Parameters:
    -----------
    protein_name : str
        Name of the protein to search for
    
    Returns:
    --------
    Dict or None
        Reference data if found, None otherwise
    """
    
    matches = frap_reference_db.search_by_protein(protein_name)
    
    if len(matches) == 0:
        return None
    
    # Return summary of matches
    return {
        'matches': matches,
        'n_matches': len(matches),
        'classes': matches['class_type'].unique().tolist(),
        'compartments': matches['cellular_compartment'].unique().tolist(),
        'deff_range': [matches['deff_um2_s'].min(), matches['deff_um2_s'].max()] if matches['deff_um2_s'].notna().any() else None,
        'mf_range': [matches['mobile_fraction_pct'].min(), matches['mobile_fraction_pct'].max()] if matches['mobile_fraction_pct'].notna().any() else None
    }

def plot_experimental_vs_reference(experimental_data: Dict, 
                                  reference_matches: pd.DataFrame,
                                  title: str = "Experimental vs Reference Data") -> go.Figure:
    """
    Create a plot comparing experimental data to reference matches.
    
    Parameters:
    -----------
    experimental_data : Dict
        Dictionary with experimental values (keys: 'deff', 'mobile_fraction', 'name')
    reference_matches : pd.DataFrame
        Reference database matches
    title : str
        Plot title
    
    Returns:
    --------
    go.Figure
        Plotly figure comparing experimental and reference data
    """
    
    fig = go.Figure()
    
    # Add reference data points
    ref_data = reference_matches[reference_matches['deff_um2_s'].notna() & reference_matches['mobile_fraction_pct'].notna()]
    
    if len(ref_data) > 0:
        fig.add_trace(go.Scatter(
            x=ref_data['deff_um2_s'],
            y=ref_data['mobile_fraction_pct'],
            mode='markers',
            marker=dict(
                size=8,
                color=ref_data['class_type'].astype('category').cat.codes,
                colorscale='Set3',
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            text=ref_data['protein_probe'],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "D_eff: %{x:.3f} µm²/s<br>"
                "Mobile Fraction: %{y:.1f}%<br>"
                "Class: %{customdata[0]}<br>"
                "Location: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=ref_data[['class_type', 'cellular_compartment']].values,
            name='Reference Data',
            showlegend=True
        ))
    
    # Add experimental data point
    if experimental_data.get('deff') is not None and experimental_data.get('mobile_fraction') is not None:
        fig.add_trace(go.Scatter(
            x=[experimental_data['deff']],
            y=[experimental_data['mobile_fraction']],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(width=2, color='darkred')
            ),
            name=experimental_data.get('name', 'Your Data'),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "D_eff: %{x:.3f} µm²/s<br>"
                "Mobile Fraction: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
            text=[experimental_data.get('name', 'Your Data')],
            showlegend=True
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Diffusion Coefficient (µm²/s)",
        yaxis_title="Mobile Fraction (%)",
        xaxis_type="log",
        height=500,
        hovermode='closest'
    )
    
    return fig

def suggest_reference_proteins(experimental_deff: Optional[float] = None,
                              experimental_mf: Optional[float] = None,
                              compartment: Optional[str] = None,
                              n_suggestions: int = 5) -> List[Dict]:
    """
    Suggest reference proteins for experimental design or comparison.
    
    Parameters:
    -----------
    experimental_deff : float, optional
        Target diffusion coefficient
    experimental_mf : float, optional
        Target mobile fraction
    compartment : str, optional
        Cellular compartment of interest
    n_suggestions : int
        Number of suggestions to return
    
    Returns:
    --------
    List[Dict]
        List of suggested reference proteins with rationale
    """
    
    suggestions = []
    
    # Get similar proteins
    similar = frap_reference_db.get_similar_proteins(
        deff=experimental_deff,
        mobile_fraction=experimental_mf,
        compartment=compartment,
        tolerance=0.4
    )
    
    # Convert to suggestions format
    for _, protein in similar.head(n_suggestions).iterrows():
        suggestion = {
            'protein': protein['protein_probe'],
            'class': protein['class_type'],
            'compartment': protein['cellular_compartment'],
            'deff': protein['deff_um2_s'],
            'mobile_fraction': protein['mobile_fraction_pct'],
            'key_finding': protein['key_finding'],
            'reference': protein['reference'],
            'rationale': f"Similar mobility parameters in {protein['cellular_compartment']}"
        }
        
        # Add specific rationale based on matching criteria
        if experimental_deff is not None and pd.notna(protein['deff_um2_s']):
            diff_ratio = abs(protein['deff_um2_s'] - experimental_deff) / experimental_deff
            if diff_ratio < 0.2:
                suggestion['rationale'] += f" (D_eff within 20%)"
        
        if experimental_mf is not None and pd.notna(protein['mobile_fraction_pct']):
            mf_diff = abs(protein['mobile_fraction_pct'] - experimental_mf)
            if mf_diff < 10:
                suggestion['rationale'] += f" (Mf within 10%)"
        
        suggestions.append(suggestion)
    
    return suggestions