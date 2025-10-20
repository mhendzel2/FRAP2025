#!/usr/bin/env python3
"""
FRAP Reference Database Module

This module contains a comprehensive database of protein mobility parameters
from published FRAP studies, providing reference values for comparison with
experimental results.

The database includes:
- Inert probes (GFP variants, dextrans)
- Structural proteins (nuclear lamins)
- Chromatin-associated factors (DNA repair, chromatin remodelers, splicing factors)
- Membrane and signaling proteins

Author: FRAP2025 Analysis Platform
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class FRAPReferenceDatabase:
    """
    Comprehensive database of protein mobility data from FRAP studies.
    
    This class provides methods to search, filter, and compare experimental
    results against published reference values.
    """
    
    def __init__(self):
        """Initialize the reference database."""
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load the comprehensive protein mobility reference database."""
        
        # Define the reference database
        reference_data = [
            # Inert Probes
            {
                'protein_probe': 'GFP Monomer',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 27,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'HeLa',
                'condition': 'Undamaged',
                'deff_um2_s': 33.3,
                'deff_sem': 3.6,
                'mobile_fraction_pct': 100,
                'residence_time_s': None,
                'key_finding': 'Benchmark for free diffusion',
                'reference': '[7]'
            },
            {
                'protein_probe': 'GFP Monomer',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 27,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'HeLa',
                'condition': 'Undamaged',
                'deff_um2_s': 28,
                'deff_sem': None,
                'mobile_fraction_pct': 97,
                'residence_time_s': None,
                'key_finding': 'Benchmark for free diffusion',
                'reference': '[31]'
            },
            {
                'protein_probe': 'GFP Dimer',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 55,
                'cellular_compartment': 'E. coli Cytoplasm',
                'cell_type': 'E. coli',
                'condition': 'Undamaged',
                'deff_um2_s': 10.9,
                'deff_sem': 1.0,
                'mobile_fraction_pct': 100,
                'residence_time_s': None,
                'key_finding': 'Diffusion decreases with size',
                'reference': '[32]'
            },
            {
                'protein_probe': 'GFP Trimer',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 83,
                'cellular_compartment': 'E. coli Cytoplasm',
                'cell_type': 'E. coli',
                'condition': 'Undamaged',
                'deff_um2_s': 8.8,
                'deff_sem': 0.6,
                'mobile_fraction_pct': 100,
                'residence_time_s': None,
                'key_finding': 'Diffusion decreases with size',
                'reference': '[32]'
            },
            {
                'protein_probe': 'GFP Tetramer',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 110,
                'cellular_compartment': 'E. coli Cytoplasm',
                'cell_type': 'E. coli',
                'condition': 'Undamaged',
                'deff_um2_s': 8.0,
                'deff_sem': 0.6,
                'mobile_fraction_pct': 100,
                'residence_time_s': None,
                'key_finding': 'Diffusion decreases with size',
                'reference': '[32]'
            },
            {
                'protein_probe': 'GFP Pentamer',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 138,
                'cellular_compartment': 'E. coli Cytoplasm',
                'cell_type': 'E. coli',
                'condition': 'Undamaged',
                'deff_um2_s': 7.1,
                'deff_sem': 0.5,
                'mobile_fraction_pct': 100,
                'residence_time_s': None,
                'key_finding': 'Diffusion decreases with size',
                'reference': '[32]'
            },
            {
                'protein_probe': '70 kDa Dextran',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 70,
                'cellular_compartment': 'Cytoplasm',
                'cell_type': 'MDCK / 3T3',
                'condition': 'Undamaged',
                'deff_um2_s': 12.5,
                'deff_sem': None,
                'mobile_fraction_pct': 80,  # >75%
                'residence_time_s': None,
                'key_finding': 'Slower than in water, highly mobile',
                'reference': '[33]'
            },
            {
                'protein_probe': '70 kDa Dextran',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 70,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'MDCK / 3T3',
                'condition': 'Undamaged',
                'deff_um2_s': 12.0,
                'deff_sem': None,
                'mobile_fraction_pct': 80,  # >75%
                'residence_time_s': None,
                'key_finding': 'Slower than in water, highly mobile',
                'reference': '[33]'
            },
            {
                'protein_probe': '500 kDa Dextran',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 500,
                'cellular_compartment': 'Cytoplasm',
                'cell_type': 'MDCK / 3T3',
                'condition': 'Undamaged',
                'deff_um2_s': 4.0,
                'deff_sem': None,
                'mobile_fraction_pct': 40,  # <50%
                'residence_time_s': None,
                'key_finding': 'Very large molecules show reduced mobile fraction',
                'reference': '[33]'
            },
            {
                'protein_probe': '500 kDa Dextran',
                'class_type': 'Inert Probe',
                'molecular_weight_kda': 500,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'MDCK / 3T3',
                'condition': 'Undamaged',
                'deff_um2_s': 4.0,
                'deff_sem': None,
                'mobile_fraction_pct': 40,  # <50%
                'residence_time_s': None,
                'key_finding': 'Very large molecules show reduced mobile fraction',
                'reference': '[33]'
            },
            
            # Structural Proteins
            {
                'protein_probe': 'WT GFP-Lamin A/C',
                'class_type': 'Nuclear Lamin',
                'molecular_weight_kda': 74,  # Approximate MW of lamin A
                'cellular_compartment': 'Nuclear Lamina',
                'cell_type': 'Mammalian',
                'condition': 'Undamaged',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': 20,  # <25%
                'residence_time_s': None,
                'key_finding': 'Lamina is a stable structure with low subunit turnover',
                'reference': '[20]'
            },
            {
                'protein_probe': 'WT GFP-Lamin A',
                'class_type': 'Nuclear Lamin',
                'molecular_weight_kda': 74,
                'cellular_compartment': 'Nuclear Lamina',
                'cell_type': 'Mammalian',
                'condition': 'Undamaged',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': 35.4,
                'key_finding': 'Establishes baseline mobility for WT protein',
                'reference': '[37]'
            },
            {
                'protein_probe': 'GFP-Lamin A S22D',
                'class_type': 'Nuclear Lamin',
                'molecular_weight_kda': 74,
                'cellular_compartment': 'Nuclear Lamina',
                'cell_type': 'Mammalian',
                'condition': 'Phosphomimetic',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': 45.4,
                'key_finding': 'Phosphomimetic mutation increases subunit exchange',
                'reference': '[37]'
            },
            {
                'protein_probe': 'GFP-Lamin A S22A',
                'class_type': 'Nuclear Lamin',
                'molecular_weight_kda': 74,
                'cellular_compartment': 'Nuclear Lamina',
                'cell_type': 'Mammalian',
                'condition': 'Phospho-deficient',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': 25.7,
                'key_finding': 'Phospho-deficient mutation stabilizes the lamina',
                'reference': '[37]'
            },
            {
                'protein_probe': 'WT GFP-Lamin A/C',
                'class_type': 'Nuclear Lamin',
                'molecular_weight_kda': 74,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'Mammalian',
                'condition': 'Undamaged',
                'deff_um2_s': 3.0,  # Average of 0.4-5.6 range
                'deff_sem': None,
                'mobile_fraction_pct': 95,  # >90%
                'residence_time_s': None,
                'key_finding': 'Nucleoplasmic pool is highly mobile and diffusive',
                'reference': '[20]'
            },
            
            # Chromatin-Associated Factors
            {
                'protein_probe': 'PARP1-GFP',
                'class_type': 'DNA Repair',
                'molecular_weight_kda': 113,
                'cellular_compartment': 'Damage Site',
                'cell_type': 'Mammalian',
                'condition': 'DNA Damage',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': 78,
                'residence_time_s': None,
                'key_finding': 'Transiently immobilized but dynamic at damage sites',
                'reference': '[43]'
            },
            {
                'protein_probe': 'Ku80',
                'class_type': 'DNA Repair',
                'molecular_weight_kda': 83,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'Mammalian',
                'condition': 'Undamaged',
                'deff_um2_s': 7.5,  # Average of 0.35-14.7 range
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': None,
                'key_finding': 'Highly dynamic with multiple mobility states',
                'reference': '[46, 47]'
            },
            {
                'protein_probe': 'BRG1',
                'class_type': 'Chromatin Remodeler',
                'molecular_weight_kda': 185,
                'cellular_compartment': 'MMTV Array',
                'cell_type': 'Mammalian',
                'condition': 'Hormone-activated',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': 3.9,
                'key_finding': 'Dynamic association with target chromatin',
                'reference': '[49]'
            },
            {
                'protein_probe': 'BRM',
                'class_type': 'Chromatin Remodeler',
                'molecular_weight_kda': 180,
                'cellular_compartment': 'MMTV Array',
                'cell_type': 'Mammalian',
                'condition': 'Hormone-activated',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': 1.95,
                'key_finding': 'More transient interaction than BRG1',
                'reference': '[49]'
            },
            {
                'protein_probe': 'BRG1 (ATPase-dead)',
                'class_type': 'Chromatin Remodeler',
                'molecular_weight_kda': 185,
                'cellular_compartment': 'MMTV Array',
                'cell_type': 'Mammalian',
                'condition': 'Hormone-activated',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': 5.5,
                'key_finding': 'ATP hydrolysis is required for efficient dissociation',
                'reference': '[49]'
            },
            {
                'protein_probe': 'Snf2H / Snf2L',
                'class_type': 'Chromatin Remodeler',
                'molecular_weight_kda': 122,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'Mammalian',
                'condition': 'Undamaged',
                'deff_um2_s': 6.0,  # Average of 5.5-6.5 range
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': 0.002,
                'key_finding': 'Extremely rapid "hit-and-run" scanning of chromatin',
                'reference': '[51]'
            },
            {
                'protein_probe': 'ASF/SF2',
                'class_type': 'Splicing Factor',
                'molecular_weight_kda': 28,
                'cellular_compartment': 'Nucleoplasm',
                'cell_type': 'Mammalian',
                'condition': 'Undamaged',
                'deff_um2_s': 0.24,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': None,
                'key_finding': 'Slow mobility due to widespread transient binding',
                'reference': '[27]'
            },
            {
                'protein_probe': 'ASF/SF2',
                'class_type': 'Splicing Factor',
                'molecular_weight_kda': 28,
                'cellular_compartment': 'Nuclear Speckle',
                'cell_type': 'Mammalian',
                'condition': 'Undamaged',
                'deff_um2_s': 0.24,
                'deff_sem': None,
                'mobile_fraction_pct': None,
                'residence_time_s': None,
                'key_finding': 'Mobility is identical inside and outside of speckles',
                'reference': '[27]'
            },
            
            # Membrane & Signaling
            {
                'protein_probe': 'EGFR-eGFP',
                'class_type': 'Membrane Protein',
                'molecular_weight_kda': 175,
                'cellular_compartment': 'Plasma Membrane',
                'cell_type': 'CHO',
                'condition': 'Unstimulated',
                'deff_um2_s': 0.067,
                'deff_sem': None,
                'mobile_fraction_pct': 86,
                'residence_time_s': None,
                'key_finding': 'Baseline lateral diffusion of monomeric/dimeric receptor',
                'reference': '[52]'
            },
            {
                'protein_probe': 'EGFR-eGFP',
                'class_type': 'Membrane Protein',
                'molecular_weight_kda': 175,
                'cellular_compartment': 'Plasma Membrane',
                'cell_type': 'CHO',
                'condition': '+ EGF',
                'deff_um2_s': 0.040,
                'deff_sem': None,
                'mobile_fraction_pct': 85,
                'residence_time_s': None,
                'key_finding': 'Ligand binding induces oligomerization, slowing diffusion',
                'reference': '[52]'
            },
            {
                'protein_probe': 'Dysferlin',
                'class_type': 'Signaling Protein',
                'molecular_weight_kda': 230,
                'cellular_compartment': 'Cytoplasm (muscle)',
                'cell_type': 'Muscle',
                'condition': 'Undamaged',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': 40,
                'residence_time_s': 330,
                'key_finding': 'Shows a large immobile fraction and slow dynamics',
                'reference': '[58]'
            },
            {
                'protein_probe': 'Fluo5N-Ca²⁺',
                'class_type': 'Signaling Molecule',
                'molecular_weight_kda': 1.2,  # Small molecule
                'cellular_compartment': 'Cytoplasm (muscle)',
                'cell_type': 'Muscle',
                'condition': 'Undamaged',
                'deff_um2_s': None,
                'deff_sem': None,
                'mobile_fraction_pct': 100,
                'residence_time_s': 3,
                'key_finding': 'Example of a small, fast-diffusing molecule',
                'reference': '[58]'
            }
        ]
        
        # Convert to DataFrame
        self.df = pd.DataFrame(reference_data)
        
        # Add computed fields
        self._add_computed_fields()
    
    def _add_computed_fields(self):
        """Add computed fields to the database."""
        
        # Size category based on molecular weight
        def get_size_category(mw):
            if mw < 30:
                return 'Small (< 30 kDa)'
            elif mw < 80:
                return 'Medium (30-80 kDa)'
            elif mw < 150:
                return 'Large (80-150 kDa)'
            else:
                return 'Very Large (> 150 kDa)'
        
        self.df['size_category'] = self.df['molecular_weight_kda'].apply(get_size_category)
        
        # Mobility category based on diffusion coefficient
        def get_mobility_category(deff):
            if pd.isna(deff):
                return 'Unknown'
            elif deff > 10:
                return 'Fast (> 10 µm²/s)'
            elif deff > 1:
                return 'Medium (1-10 µm²/s)'
            elif deff > 0.1:
                return 'Slow (0.1-1 µm²/s)'
            else:
                return 'Very Slow (< 0.1 µm²/s)'
        
        self.df['mobility_category'] = self.df['deff_um2_s'].apply(get_mobility_category)
    
    def search_by_class(self, class_type: str) -> pd.DataFrame:
        """Search database by protein class."""
        return self.df[self.df['class_type'].str.contains(class_type, case=False, na=False)]
    
    def search_by_compartment(self, compartment: str) -> pd.DataFrame:
        """Search database by cellular compartment."""
        return self.df[self.df['cellular_compartment'].str.contains(compartment, case=False, na=False)]
    
    def search_by_protein(self, protein_name: str) -> pd.DataFrame:
        """Search database by protein name."""
        return self.df[self.df['protein_probe'].str.contains(protein_name, case=False, na=False)]
    
    def get_similar_proteins(self, 
                           molecular_weight: Optional[float] = None,
                           deff: Optional[float] = None,
                           mobile_fraction: Optional[float] = None,
                           compartment: Optional[str] = None,
                           tolerance: float = 0.3) -> pd.DataFrame:
        """
        Find proteins with similar characteristics to experimental values.
        
        Parameters:
        -----------
        molecular_weight : float, optional
            Molecular weight in kDa
        deff : float, optional
            Diffusion coefficient in µm²/s
        mobile_fraction : float, optional
            Mobile fraction in %
        compartment : str, optional
            Cellular compartment
        tolerance : float
            Relative tolerance for matching (0.3 = 30%)
        
        Returns:
        --------
        pd.DataFrame
            Filtered database with similar proteins
        """
        
        result = self.df.copy()
        
        # Filter by compartment if specified
        if compartment:
            result = result[result['cellular_compartment'].str.contains(compartment, case=False, na=False)]
        
        # Filter by molecular weight if specified
        if molecular_weight is not None:
            mw_min = molecular_weight * (1 - tolerance)
            mw_max = molecular_weight * (1 + tolerance)
            result = result[
                (result['molecular_weight_kda'] >= mw_min) & 
                (result['molecular_weight_kda'] <= mw_max)
            ]
        
        # Filter by diffusion coefficient if specified
        if deff is not None:
            deff_min = deff * (1 - tolerance)
            deff_max = deff * (1 + tolerance)
            result = result[
                (result['deff_um2_s'] >= deff_min) & 
                (result['deff_um2_s'] <= deff_max) &
                (result['deff_um2_s'].notna())
            ]
        
        # Filter by mobile fraction if specified
        if mobile_fraction is not None:
            mf_min = mobile_fraction * (1 - tolerance)
            mf_max = mobile_fraction * (1 + tolerance)
            result = result[
                (result['mobile_fraction_pct'] >= mf_min) & 
                (result['mobile_fraction_pct'] <= mf_max) &
                (result['mobile_fraction_pct'].notna())
            ]
        
        return result.sort_values('deff_um2_s', na_position='last')
    
    def get_reference_ranges(self, class_type: Optional[str] = None) -> Dict:
        """
        Get statistical ranges for different protein classes.
        
        Parameters:
        -----------
        class_type : str, optional
            Specific protein class to analyze
        
        Returns:
        --------
        Dict
            Dictionary with statistical summaries
        """
        
        if class_type:
            data = self.df[self.df['class_type'].str.contains(class_type, case=False, na=False)]
        else:
            data = self.df
        
        stats = {}
        
        # Diffusion coefficient statistics
        deff_data = data['deff_um2_s'].dropna()
        if len(deff_data) > 0:
            stats['deff'] = {
                'mean': deff_data.mean(),
                'median': deff_data.median(),
                'std': deff_data.std(),
                'min': deff_data.min(),
                'max': deff_data.max(),
                'q25': deff_data.quantile(0.25),
                'q75': deff_data.quantile(0.75)
            }
        
        # Mobile fraction statistics
        mf_data = data['mobile_fraction_pct'].dropna()
        if len(mf_data) > 0:
            stats['mobile_fraction'] = {
                'mean': mf_data.mean(),
                'median': mf_data.median(),
                'std': mf_data.std(),
                'min': mf_data.min(),
                'max': mf_data.max(),
                'q25': mf_data.quantile(0.25),
                'q75': mf_data.quantile(0.75)
            }
        
        # Molecular weight statistics
        mw_data = data['molecular_weight_kda'].dropna()
        if len(mw_data) > 0:
            stats['molecular_weight'] = {
                'mean': mw_data.mean(),
                'median': mw_data.median(),
                'std': mw_data.std(),
                'min': mw_data.min(),
                'max': mw_data.max(),
                'q25': mw_data.quantile(0.25),
                'q75': mw_data.quantile(0.75)
            }
        
        return stats
    
    def plot_reference_overview(self) -> go.Figure:
        """Create an overview plot of the reference database."""
        
        # Filter data with valid diffusion coefficients
        plot_data = self.df[self.df['deff_um2_s'].notna()].copy()
        
        if len(plot_data) == 0:
            # Create empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No diffusion coefficient data available for plotting",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create scatter plot
        fig = px.scatter(
            plot_data,
            x='molecular_weight_kda',
            y='deff_um2_s',
            color='class_type',
            size='mobile_fraction_pct',
            hover_data=['protein_probe', 'cellular_compartment', 'cell_type', 'condition'],
            labels={
                'molecular_weight_kda': 'Molecular Weight (kDa)',
                'deff_um2_s': 'Diffusion Coefficient (µm²/s)',
                'class_type': 'Protein Class',
                'mobile_fraction_pct': 'Mobile Fraction (%)'
            },
            title='FRAP Reference Database: Protein Mobility Overview',
            log_y=True
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        return fig
    
    def plot_class_comparison(self, parameter: str = 'deff_um2_s') -> go.Figure:
        """Create a comparison plot by protein class."""
        
        valid_params = ['deff_um2_s', 'mobile_fraction_pct', 'molecular_weight_kda']
        if parameter not in valid_params:
            parameter = 'deff_um2_s'
        
        # Filter data
        plot_data = self.df[self.df[parameter].notna()].copy()
        
        if len(plot_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No {parameter} data available for plotting",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create box plot
        fig = px.box(
            plot_data,
            x='class_type',
            y=parameter,
            points="all",
            hover_data=['protein_probe', 'cellular_compartment'],
            title=f'Distribution of {parameter.replace("_", " ").title()} by Protein Class'
        )
        
        if parameter == 'deff_um2_s':
            fig.update_layout(yaxis_type="log", yaxis_title="Diffusion Coefficient (µm²/s)")
        elif parameter == 'mobile_fraction_pct':
            fig.update_layout(yaxis_title="Mobile Fraction (%)")
        elif parameter == 'molecular_weight_kda':
            fig.update_layout(yaxis_title="Molecular Weight (kDa)")
        
        fig.update_layout(xaxis_title="Protein Class")
        fig.update_layout(height=500, xaxis_tickangle=-45)
        
        return fig
    
    def compare_experimental_to_reference(self, 
                                        experimental_deff: Optional[float] = None,
                                        experimental_mf: Optional[float] = None,
                                        experimental_mw: Optional[float] = None,
                                        compartment: Optional[str] = None) -> Dict:
        """
        Compare experimental values to reference database.
        
        Returns analysis and recommendations based on database comparisons.
        """
        
        results = {
            'closest_matches': None,
            'class_percentiles': {},
            'interpretation': "",
            'recommendations': []
        }
        
        # Find closest matches
        if any(v is not None for v in [experimental_deff, experimental_mf, experimental_mw]):
            closest = self.get_similar_proteins(
                molecular_weight=experimental_mw,
                deff=experimental_deff,
                mobile_fraction=experimental_mf,
                compartment=compartment,
                tolerance=0.5  # 50% tolerance for broad matching
            )
            
            if len(closest) > 0:
                results['closest_matches'] = closest.head(10)
        
        # Calculate percentiles within each class
        if experimental_deff is not None:
            for class_type in self.df['class_type'].unique():
                class_data = self.df[
                    (self.df['class_type'] == class_type) & 
                    (self.df['deff_um2_s'].notna())
                ]
                
                if len(class_data) > 0:
                    percentile = (class_data['deff_um2_s'] < experimental_deff).mean() * 100
                    results['class_percentiles'][class_type] = {
                        'deff_percentile': percentile,
                        'n_samples': len(class_data)
                    }
        
        # Generate interpretation
        interpretation_parts = []
        
        if experimental_deff is not None:
            if experimental_deff > 10:
                interpretation_parts.append(
                    f"The diffusion coefficient ({experimental_deff:.2f} µm²/s) is in the fast mobility range, "
                    "similar to inert probes and highly mobile proteins."
                )
            elif experimental_deff > 1:
                interpretation_parts.append(
                    f"The diffusion coefficient ({experimental_deff:.2f} µm²/s) indicates medium mobility, "
                    "typical of many nuclear proteins with transient interactions."
                )
            elif experimental_deff > 0.1:
                interpretation_parts.append(
                    f"The diffusion coefficient ({experimental_deff:.2f} µm²/s) suggests slow mobility, "
                    "indicating significant binding or structural constraints."
                )
            else:
                interpretation_parts.append(
                    f"The diffusion coefficient ({experimental_deff:.2f} µm²/s) is very slow, "
                    "suggesting strong binding or incorporation into stable structures."
                )
        
        results['interpretation'] = " ".join(interpretation_parts)
        
        # Generate recommendations
        if results['closest_matches'] is not None and len(results['closest_matches']) > 0:
            top_match = results['closest_matches'].iloc[0]
            results['recommendations'].append(
                f"Your protein shows similar mobility to {top_match['protein_probe']} "
                f"({top_match['class_type']}), which {top_match['key_finding'].lower()}."
            )
        
        if experimental_deff is not None and experimental_deff < 1:
            results['recommendations'].append(
                "Consider investigating potential binding partners or structural associations "
                "that could explain the reduced mobility."
            )
        
        if experimental_mf is not None and experimental_mf < 70:
            results['recommendations'].append(
                "The reduced mobile fraction suggests a significant population of immobilized proteins. "
                "Consider analyzing binding kinetics or structural integration."
            )
        
        return results


# Global instance for easy access
frap_reference_db = FRAPReferenceDatabase()


def display_reference_database_ui():
    """Display the reference database interface in Streamlit."""
    
    st.header("📚 FRAP Reference Database")
    
    st.markdown("""
    This comprehensive database contains protein mobility parameters from published FRAP studies, 
    providing reference values for comparison with your experimental results.
    """)
    
    # Database overview tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search & Browse", 
        "📊 Overview Plots", 
        "🎯 Compare to Reference", 
        "📋 Full Database"
    ])
    
    with tab1:
        st.subheader("Search Reference Database")
        
        # Search options
        search_type = st.selectbox(
            "Search by:",
            ["Protein Name", "Protein Class", "Cellular Compartment", "Similar Properties"]
        )
        
        if search_type == "Protein Name":
            protein_query = st.text_input("Enter protein name:", placeholder="e.g., GFP, Lamin, PARP")
            if protein_query:
                results = frap_reference_db.search_by_protein(protein_query)
                if len(results) > 0:
                    st.dataframe(results[['protein_probe', 'class_type', 'molecular_weight_kda', 
                                       'deff_um2_s', 'mobile_fraction_pct', 'key_finding']])
                else:
                    st.warning("No matching proteins found.")
        
        elif search_type == "Protein Class":
            class_options = frap_reference_db.df['class_type'].unique()
            selected_class = st.selectbox("Select protein class:", class_options)
            results = frap_reference_db.search_by_class(selected_class)
            st.dataframe(results[['protein_probe', 'cellular_compartment', 'molecular_weight_kda', 
                                'deff_um2_s', 'mobile_fraction_pct', 'key_finding']])
        
        elif search_type == "Cellular Compartment":
            compartment_options = frap_reference_db.df['cellular_compartment'].unique()
            selected_compartment = st.selectbox("Select cellular compartment:", compartment_options)
            results = frap_reference_db.search_by_compartment(selected_compartment)
            st.dataframe(results[['protein_probe', 'class_type', 'molecular_weight_kda', 
                                'deff_um2_s', 'mobile_fraction_pct', 'key_finding']])
        
        elif search_type == "Similar Properties":
            st.markdown("**Find proteins with similar characteristics:**")
            
            col1, col2 = st.columns(2)
            with col1:
                search_mw = st.number_input("Molecular Weight (kDa):", min_value=0.0, value=None, 
                                          placeholder="Optional")
                search_deff = st.number_input("Diffusion Coefficient (µm²/s):", min_value=0.0, value=None,
                                            placeholder="Optional")
            
            with col2:
                search_mf = st.number_input("Mobile Fraction (%):", min_value=0.0, max_value=100.0, 
                                          value=None, placeholder="Optional")
                search_compartment = st.text_input("Compartment:", placeholder="Optional")
            
            tolerance = st.slider("Search tolerance (±%):", 10, 100, 30) / 100
            
            if st.button("Search Similar Proteins"):
                results = frap_reference_db.get_similar_proteins(
                    molecular_weight=search_mw,
                    deff=search_deff,
                    mobile_fraction=search_mf,
                    compartment=search_compartment,
                    tolerance=tolerance
                )
                
                if len(results) > 0:
                    st.success(f"Found {len(results)} similar proteins:")
                    st.dataframe(results[['protein_probe', 'class_type', 'cellular_compartment',
                                        'molecular_weight_kda', 'deff_um2_s', 'mobile_fraction_pct', 
                                        'key_finding']])
                else:
                    st.warning("No similar proteins found. Try adjusting the search criteria or tolerance.")
    
    with tab2:
        st.subheader("Database Overview")
        
        # Overview plot
        fig_overview = frap_reference_db.plot_reference_overview()
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Class comparison
        st.subheader("Compare by Protein Class")
        comparison_param = st.selectbox(
            "Parameter to compare:",
            ["deff_um2_s", "mobile_fraction_pct", "molecular_weight_kda"],
            format_func=lambda x: {
                'deff_um2_s': 'Diffusion Coefficient',
                'mobile_fraction_pct': 'Mobile Fraction',
                'molecular_weight_kda': 'Molecular Weight'
            }[x]
        )
        
        fig_class = frap_reference_db.plot_class_comparison(comparison_param)
        st.plotly_chart(fig_class, use_container_width=True)
        
        # Statistics summary
        st.subheader("Database Statistics")
        stats = frap_reference_db.get_reference_ranges()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Proteins", len(frap_reference_db.df))
            if 'deff' in stats:
                st.metric("Proteins with Deff", len(frap_reference_db.df[frap_reference_db.df['deff_um2_s'].notna()]))
        
        with col2:
            st.metric("Protein Classes", frap_reference_db.df['class_type'].nunique())
            if 'mobile_fraction' in stats:
                st.metric("Proteins with Mf", len(frap_reference_db.df[frap_reference_db.df['mobile_fraction_pct'].notna()]))
        
        with col3:
            st.metric("Compartments", frap_reference_db.df['cellular_compartment'].nunique())
            st.metric("Cell Types", frap_reference_db.df['cell_type'].nunique())
    
    with tab3:
        st.subheader("Compare Your Results to Reference Data")
        
        st.markdown("""
        Enter your experimental parameters to compare against the reference database 
        and receive interpretation and recommendations.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            exp_deff = st.number_input("Your Diffusion Coefficient (µm²/s):", min_value=0.0, 
                                     value=None, placeholder="Optional")
            exp_mf = st.number_input("Your Mobile Fraction (%):", min_value=0.0, max_value=100.0,
                                   value=None, placeholder="Optional")
        
        with col2:
            exp_mw = st.number_input("Protein Molecular Weight (kDa):", min_value=0.0,
                                   value=None, placeholder="Optional")
            exp_compartment = st.text_input("Cellular Compartment:", placeholder="e.g., Nucleoplasm")
        
        if st.button("Compare to Reference Database"):
            if any(v is not None for v in [exp_deff, exp_mf, exp_mw]):
                comparison = frap_reference_db.compare_experimental_to_reference(
                    experimental_deff=exp_deff,
                    experimental_mf=exp_mf,
                    experimental_mw=exp_mw,
                    compartment=exp_compartment
                )
                
                # Display interpretation
                if comparison['interpretation']:
                    st.subheader("📊 Interpretation")
                    st.info(comparison['interpretation'])
                
                # Display recommendations
                if comparison['recommendations']:
                    st.subheader("💡 Recommendations")
                    for i, rec in enumerate(comparison['recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
                
                # Display closest matches
                if comparison['closest_matches'] is not None and len(comparison['closest_matches']) > 0:
                    st.subheader("🎯 Closest Matches in Database")
                    st.dataframe(
                        comparison['closest_matches'][
                            ['protein_probe', 'class_type', 'cellular_compartment', 
                             'molecular_weight_kda', 'deff_um2_s', 'mobile_fraction_pct', 
                             'key_finding', 'reference']
                        ]
                    )
                
                # Display percentile rankings
                if comparison['class_percentiles']:
                    st.subheader("📈 Your Protein Compared to Each Class")
                    percentile_data = []
                    for class_type, stats in comparison['class_percentiles'].items():
                        if stats['n_samples'] >= 2:  # Only show classes with sufficient data
                            percentile_data.append({
                                'Protein Class': class_type,
                                'Your Percentile': f"{stats['deff_percentile']:.1f}%",
                                'Reference Samples': stats['n_samples']
                            })
                    
                    if percentile_data:
                        st.dataframe(pd.DataFrame(percentile_data))
                        st.caption("Percentile shows how your diffusion coefficient compares within each protein class (higher = faster than more proteins in that class)")
            
            else:
                st.warning("Please enter at least one experimental parameter to compare.")
    
    with tab4:
        st.subheader("Complete Reference Database")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_class = st.selectbox(
                "Filter by class:",
                ["All"] + list(frap_reference_db.df['class_type'].unique())
            )
        
        with col2:
            filter_compartment = st.selectbox(
                "Filter by compartment:",
                ["All"] + list(frap_reference_db.df['cellular_compartment'].unique())
            )
        
        with col3:
            show_columns = st.multiselect(
                "Show columns:",
                frap_reference_db.df.columns.tolist(),
                default=['protein_probe', 'class_type', 'molecular_weight_kda', 
                        'cellular_compartment', 'deff_um2_s', 'mobile_fraction_pct', 
                        'key_finding']
            )
        
        # Apply filters
        filtered_df = frap_reference_db.df.copy()
        
        if filter_class != "All":
            filtered_df = filtered_df[filtered_df['class_type'] == filter_class]
        
        if filter_compartment != "All":
            filtered_df = filtered_df[filtered_df['cellular_compartment'] == filter_compartment]
        
        # Display filtered data
        if show_columns:
            st.dataframe(filtered_df[show_columns], use_container_width=True)
        
        # Export option
        if st.button("📥 Download Database as CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="frap_reference_database.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    # Test the database
    db = FRAPReferenceDatabase()
    print(f"Database loaded with {len(db.df)} entries")
    print(f"Protein classes: {db.df['class_type'].unique()}")
    
    # Test search functionality
    gfp_results = db.search_by_protein("GFP")
    print(f"\nFound {len(gfp_results)} GFP-related entries")
    
    # Test similarity search
    similar = db.get_similar_proteins(molecular_weight=30, deff=5.0, tolerance=0.5)
    print(f"\nFound {len(similar)} similar proteins")