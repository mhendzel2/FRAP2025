# Streamlit UI Integration for Advanced Group Fitting

## Location
**File:** `streamlit_frap_final_clean.py`  
**Tab:** Tab 2 - Multi-Group Comparison  
**Section:** After holistic comparison section

---

## Code to Add

### Step 1: Add UI Controls

Insert after the holistic comparison section (around line 2900):

```python
# Advanced Curve Fitting Section
st.markdown("---")
st.markdown("### 🔬 Advanced Curve Fitting Comparison")

st.markdown("""
**Fit sophisticated biophysical models to mean recovery profiles**

This feature fits advanced models (anomalous diffusion, reaction-diffusion) to the 
averaged recovery curves for each group, enabling mechanistic comparison of FRAP kinetics.

**Use when:**
- Want to understand diffusion regime (normal vs. anomalous)
- Need to separate diffusion from binding contributions
- Comparing mechanistic differences between conditions
""")

# Check for lmfit
try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    st.warning("⚠️ Advanced fitting requires lmfit. Install with: `pip install lmfit`")

if LMFIT_AVAILABLE and len(selected_groups_holistic) == 2:
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        enable_advanced_fitting = st.checkbox(
            "Enable advanced curve fitting",
            value=False,
            help="Fit sophisticated models to mean recovery profiles"
        )
    
    with col_adv2:
        if enable_advanced_fitting:
            advanced_model = st.selectbox(
                "Model selection",
                options=['all', 'anomalous', 'reaction_diffusion_simple', 'reaction_diffusion_full'],
                index=0,
                help="'all' tries all models and selects best by AIC"
            )
            
            model_descriptions = {
                'all': 'Try all models, select best (recommended)',
                'anomalous': 'Anomalous diffusion (stretched exponential)',
                'reaction_diffusion_simple': 'Reaction-diffusion (simple)',
                'reaction_diffusion_full': 'Reaction-diffusion (explicit k_on/k_off)'
            }
            st.caption(model_descriptions[advanced_model])
    
    if enable_advanced_fitting:
        bleach_radius_input = st.number_input(
            "Bleach spot radius (μm)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Radius of bleached spot in micrometers"
        )
        
        if st.button("Run Advanced Fitting", type="primary", key="advanced_fitting_btn"):
            with st.spinner("Fitting advanced models to mean recovery profiles..."):
                try:
                    # Get group data
                    group1_name, group2_name = selected_groups_holistic
                    
                    # Prepare data dictionaries
                    group1_files = dm.groups[group1_name]['files']
                    group2_files = dm.groups[group2_name]['files']
                    
                    group1_data_raw = {fp: dm.files[fp] for fp in group1_files if fp in dm.files}
                    group2_data_raw = {fp: dm.files[fp] for fp in group2_files if fp in dm.files}
                    
                    # Run comparison with advanced fitting
                    advanced_comparison = compare_recovery_profiles(
                        group1_data_raw,
                        group2_data_raw,
                        group1_name=group1_name,
                        group2_name=group2_name,
                        use_advanced_fitting=True,
                        bleach_radius_um=bleach_radius_input,
                        advanced_model=advanced_model
                    )
                    
                    # Check if successful
                    if 'advanced_fitting' in advanced_comparison and advanced_comparison['advanced_fitting']['success']:
                        adv = advanced_comparison['advanced_fitting']
                        
                        st.success("✓ Advanced fitting completed successfully")
                        
                        # Model info
                        st.markdown("#### Model Information")
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.metric("Model Used", adv['model_used'].replace('_', ' ').title())
                        with col_m2:
                            st.metric(f"R² ({group1_name})", f"{adv['r2_group1']:.4f}")
                        with col_m3:
                            st.metric(f"R² ({group2_name})", f"{adv['r2_group2']:.4f}")
                        
                        # Plot fitted curves
                        st.markdown("#### Fitted Curves")
                        from frap_plots import FRAPPlots
                        fig_advanced = FRAPPlots.plot_advanced_group_comparison(
                            advanced_comparison,
                            height=500
                        )
                        if fig_advanced:
                            st.plotly_chart(fig_advanced, use_container_width=True)
                        
                        # Parameter comparison
                        st.markdown("#### Parameter Comparison")
                        
                        if 'parameter_comparison' in adv and adv['parameter_comparison']:
                            param_data = []
                            for param, data in adv['parameter_comparison'].items():
                                param_data.append({
                                    'Parameter': param.replace('_', ' ').title(),
                                    group1_name: f"{data[group1_name]:.4f}",
                                    group2_name: f"{data[group2_name]:.4f}",
                                    'Fold Change': f"{data['fold_change']:.3f}x",
                                    'Percent Change': f"{data['percent_change']:+.1f}%"
                                })
                            
                            param_df = pd.DataFrame(param_data)
                            st.dataframe(param_df, use_container_width=True, hide_index=True)
                        
                        # Parameter comparison plot
                        fig_params = FRAPPlots.plot_parameter_comparison(
                            advanced_comparison,
                            height=400
                        )
                        if fig_params:
                            st.plotly_chart(fig_params, use_container_width=True)
                        
                        # Biological interpretation
                        st.markdown("#### Biological Interpretation")
                        if 'interpretation' in adv:
                            st.markdown(adv['interpretation'])
                        
                        # Download results
                        st.markdown("#### Export Results")
                        
                        # Prepare export data
                        export_data = {
                            'model_info': {
                                'model_name': adv['model_used'],
                                'r2_group1': adv['r2_group1'],
                                'r2_group2': adv['r2_group2'],
                            },
                            'parameters': adv.get('parameter_comparison', {}),
                            'metrics': adv.get('metric_comparison', {}),
                            'interpretation': adv.get('interpretation', '')
                        }
                        
                        import json
                        export_json = json.dumps(export_data, indent=2, default=str)
                        
                        st.download_button(
                            label="Download Fitting Results (JSON)",
                            data=export_json,
                            file_name=f"advanced_fitting_{group1_name}_vs_{group2_name}.json",
                            mime="application/json"
                        )
                    
                    else:
                        error_msg = advanced_comparison.get('advanced_fitting', {}).get('error', 'Unknown error')
                        st.error(f"❌ Advanced fitting failed: {error_msg}")
                
                except Exception as e:
                    st.error(f"Error in advanced fitting: {e}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

elif LMFIT_AVAILABLE and len(selected_groups_holistic) != 2:
    st.info("ℹ️ Advanced fitting comparison requires exactly 2 groups to be selected.")
```

---

## Visual Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🔬 Advanced Curve Fitting Comparison                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Description text]                                          │
│                                                              │
│  ☐ Enable advanced curve fitting                            │
│  [Model selection dropdown: all ▼]                          │
│  Bleach spot radius (μm): [1.0  ]                           │
│                                                              │
│  [Run Advanced Fitting]                                      │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  ✓ Advanced fitting completed successfully                  │
│                                                              │
│  Model Information                                           │
│  ┌──────────────┬─────────────┬─────────────┐              │
│  │ Model Used   │ R² (WT)     │ R² (Mutant) │              │
│  │ Anomalous... │ 0.9994      │ 0.9992      │              │
│  └──────────────┴─────────────┴─────────────┘              │
│                                                              │
│  Fitted Curves                                               │
│  [Interactive Plotly plot showing data + fits]              │
│                                                              │
│  Parameter Comparison                                        │
│  ┌─────────┬──────┬────────┬─────────┬──────────┐         │
│  │Parameter│  WT  │ Mutant │FoldChg  │ % Change │         │
│  ├─────────┼──────┼────────┼─────────┼──────────┤         │
│  │Beta     │0.960 │ 0.607  │ 0.63x   │  -36.7%  │         │
│  │Tau      │2.981 │ 4.975  │ 1.67x   │  +66.9%  │         │
│  └─────────┴──────┴────────┴─────────┴──────────┘         │
│                                                              │
│  [Bar chart comparing parameters]                           │
│                                                              │
│  Biological Interpretation                                   │
│  Mutant shows more hindered diffusion...                    │
│  [Full interpretation text]                                 │
│                                                              │
│  [Download Fitting Results (JSON)]                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Error Handling

The code includes comprehensive error handling:

1. **Missing lmfit:**
   ```python
   st.warning("⚠️ Advanced fitting requires lmfit...")
   ```

2. **Wrong number of groups:**
   ```python
   st.info("ℹ️ Advanced fitting requires exactly 2 groups...")
   ```

3. **Fitting failure:**
   ```python
   st.error(f"❌ Advanced fitting failed: {error_msg}")
   ```

4. **General exceptions:**
   ```python
   with st.expander("Show error details"):
       st.code(traceback.format_exc())
   ```

---

## User Experience

### Workflow
1. User selects 2 groups for comparison
2. User enables advanced fitting checkbox
3. User selects model (or uses 'all' for automatic selection)
4. User enters bleach spot radius
5. User clicks "Run Advanced Fitting"
6. System fits models to both group mean profiles
7. Results displayed with plots and interpretation
8. User can download results as JSON

### Feedback
- Spinner during fitting: "Fitting advanced models..."
- Success message when complete
- Clear error messages if failed
- Expandable error details for debugging

---

## Testing

After integration, test with:

1. **Two groups selected:**
   - Should show advanced fitting section
   - Should allow model selection
   - Should run fitting successfully

2. **One or three groups:**
   - Should show info message
   - Advanced fitting section disabled

3. **lmfit not installed:**
   - Should show warning message
   - Feature disabled gracefully

4. **Bad data:**
   - Should catch errors
   - Should display error message

---

## Dependencies

Ensure these imports at top of file:

```python
import json
from frap_group_comparison import compare_recovery_profiles
from frap_plots import FRAPPlots
```

---

## Benefits

1. **Mechanistic insights** - Understand diffusion regime and binding kinetics
2. **Visual comparison** - See fitted curves overlaid on data
3. **Quantitative metrics** - Fold changes and percent changes
4. **Biological interpretation** - Automated narrative explanation
5. **Export capability** - Download results for reports

---

## Maintenance

- Update model descriptions if new models added
- Adjust default bleach_radius if needed
- Expand interpretation section as needed
- Add more export formats (CSV, Excel) if requested

---

**Integration Status:** Ready to implement  
**Estimated Time:** 30 minutes  
**Testing Required:** UI testing with real data  
**Documentation:** Complete
