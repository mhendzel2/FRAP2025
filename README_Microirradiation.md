# Microirradiation Analysis Platform

A comprehensive analysis platform for laser microirradiation experiments built on the robust FRAP2025 infrastructure.

## 🎯 Key Features

### 1. Protein Recruitment Kinetics Analysis
- Quantifies protein accumulation at DNA damage sites
- Multiple kinetic models: single/double exponential, sigmoidal
- Calculates recruitment rates, half-times, and amplitudes
- Statistical model selection using AIC/BIC criteria

### 2. Chromatin Decondensation Measurement  
- Tracks ROI expansion over time (damage-induced chromatin relaxation)
- Multiple expansion models: exponential, linear, power-law
- Adaptive mask generation for accurate measurements
- Accounts for ROI expansion in intensity analysis

### 3. Combined Microirradiation + Photobleaching Analysis
- Handles experiments with both damage induction and photobleaching
- Separates recruitment and recovery phases
- Comprehensive kinetic analysis of complex experiments
- Leverages existing FRAP analysis infrastructure

### 4. Advanced Image Analysis
- Direct analysis of microscopy image stacks
- Automated damage site detection
- Dynamic ROI tracking with expansion
- Adaptive masking to handle changing ROI geometry
- Background correction and intensity extraction

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the microirradiation analysis app
streamlit run streamlit_microirradiation.py --server.headless true --server.port 5001
```

### Usage Examples

#### 1. Data File Analysis
```python
from microirradiation_core import analyze_recruitment_kinetics, analyze_roi_expansion
import pandas as pd

# Load your data
df = pd.read_csv('your_microirradiation_data.csv')
time = df['time'].values
intensity = df['intensity'].values
roi_area = df['roi_area'].values  # Optional

# Analyze recruitment kinetics
recruitment_results = analyze_recruitment_kinetics(
    time, intensity, 
    damage_frame=5,  # Frame when microirradiation occurred
    models=['single_exp', 'double_exp', 'sigmoidal']
)

# Analyze ROI expansion
expansion_results = analyze_roi_expansion(
    time, roi_area,
    damage_frame=5,
    models=['exponential', 'linear', 'power_law']
)

print(f"Best recruitment model: {recruitment_results['best_model']}")
print(f"Recruitment rate: {recruitment_results['best_fit']['rate']:.4f} s⁻¹")
print(f"Best expansion model: {expansion_results['best_model']}")
```

#### 2. Image Stack Analysis
```python
from microirradiation_image_analysis import MicroirradiationImageAnalyzer

# Initialize analyzer
analyzer = MicroirradiationImageAnalyzer()

# Load image stack
analyzer.load_image_stack('your_timelapse.tif')
analyzer.pixel_size = 0.1  # µm per pixel
analyzer.time_interval = 2.0  # seconds per frame

# Detect damage site (automatic or manual)
damage_coords = analyzer.detect_damage_site(
    frame_range=(0, 5),
    detection_method='intensity_change'
)

# Track ROI expansion
expansion_data = analyzer.track_roi_expansion(
    initial_radius=2.0,  # µm
    method='threshold_based'
)

# Generate adaptive masks and extract intensities
adaptive_masks = analyzer.generate_adaptive_masks(expansion_data)
intensity_data = analyzer.extract_adaptive_intensities(adaptive_masks)
```

#### 3. Combined Analysis
```python
from microirradiation_core import analyze_combined_experiment

# For experiments with both microirradiation and photobleaching
result = analyze_combined_experiment(
    time, intensity,
    damage_frame=10,   # Microirradiation frame
    bleach_frame=50,   # Photobleaching frame
    roi_areas=roi_areas  # Optional ROI area data
)

print(f"Recruitment rate: {result.recruitment_rate:.4f} s⁻¹")
print(f"Expansion rate: {result.expansion_rate:.4f}")
print(f"Is combined analysis: {result.is_combined_analysis}")
```

## 📊 Data Format Requirements

### For File-Based Analysis

**Required columns:**
- `time`: Time points in seconds
- `intensity`: Fluorescence intensity at damage ROI

**Optional columns:**
- `roi_area`: ROI area measurements in µm²
- `background`: Background intensity for correction

Example CSV format:
```csv
time,intensity,roi_area,background
0.0,52.3,10.2,45.1
2.0,58.7,10.8,44.9
4.0,67.2,11.5,45.2
6.0,78.1,12.3,45.0
...
```

### For Image Analysis
- **Format**: TIFF image stacks (time-lapse)
- **Calibration**: Known pixel size (µm/pixel) and time interval (s/frame)
- **Quality**: Clear damage site visibility with good signal-to-noise ratio
- **Duration**: Sufficient pre- and post-damage time points

## 🔬 Analysis Models

### Recruitment Kinetics Models

1. **Single Exponential**
   ```
   I(t) = baseline + amplitude × (1 - e^(-rate×t))
   ```
   - Best for simple recruitment processes
   - Parameters: baseline, amplitude, rate

2. **Double Exponential**
   ```
   I(t) = baseline + amp1×(1 - e^(-rate1×t)) + amp2×(1 - e^(-rate2×t))
   ```
   - For complex recruitment with fast and slow components
   - Parameters: baseline, amp1, rate1, amp2, rate2

3. **Sigmoidal**
   ```
   I(t) = baseline + amplitude / (1 + e^(-rate×(t-lag_time)))
   ```
   - For recruitment with lag phase
   - Parameters: baseline, amplitude, rate, lag_time

### ROI Expansion Models

1. **Exponential Expansion**
   ```
   Area(t) = initial_size + max_expansion × (1 - e^(-rate×t))
   ```
   - Saturating expansion

2. **Linear Expansion**
   ```
   Area(t) = initial_size + rate × t
   ```
   - Constant expansion rate

3. **Power Law Expansion**
   ```
   Area(t) = initial_size + coefficient × t^exponent
   ```
   - Non-linear expansion dynamics

## 📈 Key Output Parameters

### Recruitment Kinetics
- **Recruitment Rate (k)**: Speed of protein accumulation (s⁻¹)
- **Half-time (t₁/₂)**: Time to reach 50% of maximum recruitment (s)
- **Amplitude**: Maximum recruitment level above baseline
- **Baseline**: Pre-damage intensity level
- **R²**: Goodness of fit (0-1, closer to 1 is better)
- **AIC**: Akaike Information Criterion (lower values indicate better model)

### ROI Expansion
- **Initial Area**: ROI size at time of damage (µm²)
- **Expansion Rate**: Speed of ROI growth
- **Maximum Expansion**: Total expansion amount (µm²)
- **Expansion Factor**: Final size / initial size ratio
- **Half-time**: Time to reach 50% of maximum expansion (s)

### Combined Analysis
- **Pre-bleach Recruitment**: Recruitment before photobleaching
- **Post-bleach Recovery**: Recovery dynamics after photobleaching
- **Phase Separation**: Clear distinction between recruitment and recovery

## 🛠️ Advanced Features

### Adaptive Masking
- ROI masks automatically adjust to expansion
- Prevents measurement artifacts from changing ROI geometry
- Essential for accurate recruitment kinetics in expanding regions

### Background Correction
- Automatic background subtraction using annular regions
- Compensation for illumination drift and cellular movement
- Improved signal-to-noise ratio

### Statistical Analysis
- Model selection using information criteria (AIC, BIC)
- Goodness-of-fit assessment (R², residual analysis)
- Multi-experiment comparison with statistical testing
- Outlier detection and handling

### Visualization
- Interactive plots with Plotly
- Overlay of data and fitted models
- Parameter confidence intervals
- Multi-experiment comparison plots

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_microirradiation.py
```

Tests include:
- Synthetic data validation
- Model fitting accuracy
- Image analysis pipeline
- Edge case handling
- Performance benchmarks

## 📚 Scientific Background

### Laser Microirradiation
Laser microirradiation creates localized DNA damage, triggering cellular repair responses. This technique allows studying:
- DNA damage response pathways
- Protein recruitment to damage sites
- Chromatin dynamics during repair
- Repair kinetics and efficiency

### Key Biological Processes
1. **Protein Recruitment**: DNA damage proteins accumulate at sites of damage
2. **Chromatin Decondensation**: Damaged chromatin relaxes to facilitate repair access
3. **Repair Dynamics**: Complex multi-phase process with distinct kinetics

### Analysis Considerations
- **Temporal Resolution**: Balance between time resolution and photobleaching
- **Spatial Resolution**: ROI size affects measurement accuracy
- **Damage Extent**: Controlled damage levels prevent artifacts
- **Cellular Context**: Cell cycle stage and type affect responses

## 🔧 Troubleshooting

### Common Issues

1. **Damage Site Not Detected**
   - Try manual selection in the interface
   - Adjust detection threshold parameters
   - Check image quality and contrast

2. **Poor Model Fits**
   - Verify data quality (sufficient signal-to-noise)
   - Try different kinetic models
   - Check for systematic artifacts (drift, bleaching)

3. **No ROI Expansion Detected**
   - Verify that damage actually causes chromatin decondensation
   - Check ROI tracking parameters
   - Ensure sufficient spatial resolution

4. **Noisy Results**
   - Increase averaging/smoothing
   - Improve imaging conditions
   - Check for systematic noise sources

### Best Practices

1. **Experimental Design**
   - Include adequate pre-damage baseline
   - Use consistent imaging conditions
   - Control laser power to minimize artifacts
   - Include appropriate controls

2. **Data Collection**
   - Sufficient temporal resolution (not too sparse)
   - Good signal-to-noise ratio
   - Minimal photobleaching during acquisition
   - Proper calibration (pixel size, time intervals)

3. **Analysis**
   - Validate automated detection results
   - Compare multiple kinetic models
   - Check fit quality and residuals
   - Use biological replicates for statistics

## 🤝 Integration with FRAP2025

This microirradiation platform leverages the robust infrastructure of FRAP2025:
- **Core Analysis**: Mathematical framework and curve fitting
- **Image Processing**: Advanced image analysis capabilities  
- **UI Framework**: Streamlit-based interface design
- **Data Management**: File handling and organization
- **Reporting**: PDF and Excel export functionality
- **Statistical Tools**: Outlier detection and comparison methods

The platforms can be used together for comprehensive studies combining FRAP and microirradiation techniques.

## 📄 Citation

If you use this platform in your research, please cite:
```
Microirradiation Analysis Platform (2024)
Built on FRAP2025 infrastructure
```

## 📞 Support

For technical issues or scientific questions:
- Check the troubleshooting section above
- Review the comprehensive documentation
- Examine the test suite for usage examples
- Ensure proper experimental design and data quality

---

*Built with ❤️ for the DNA damage response research community*