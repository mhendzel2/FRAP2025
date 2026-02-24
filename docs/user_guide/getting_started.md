# Getting Started

1. Install dependencies.
   - `pip install -e .[dev]`
2. Launch the Streamlit UI.
   - `streamlit run app.py --server.port 5000`
3. Load sample input data.
   - Use files from `sample_data/`.
4. Choose normalization mode explicitly.
   - `simple`, `double`, or `full_scale`.
5. Run fitting and export outputs.
   - Review fit quality flags before interpretation.

## Release Notes 0.4.0

- Advanced analysis modules added:
   - Population Wasserstein comparison with permutation testing.
   - Condensate capillary-wave spectroscopy with relative surface-tension estimation.
   - HDP-HMM-style Bayesian SPT analyzer with optional Pyro backend.
   - Physics-informed FRAP PINN hooks with classical fallback.
   - Unified deformation-field interface with Farneback and optional RAFT backend.
- Installation workflow consolidated:
   - Advanced extras install: `pip install -e .[advanced]`.
   - Full install with all extras: `pip install -e .[dev,db,viz,advanced]`.
- Graceful degradation is preserved:
   - If optional heavy dependencies are unavailable, supported classical/fallback methods run automatically.
