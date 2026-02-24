# Diffusion Formula Audit Result

## Final Convention (Verified)

The diffusion coefficient mapping depends on parameterization:

1. If fitting an exponential time constant `tau_s`:
   - `D_um2_per_s = w_um^2 / (4 * tau_s)`
   - equivalently `D_um2_per_s = w_um^2 * k_s / 4` with `k_s = 1/tau_s`
2. If converting from half-time `t_half_s`:
   - `D_um2_per_s = w_um^2 * ln(2) / (4 * t_half_s)`
3. Soumpasis half-time approximation:
   - `D_um2_per_s ~= 0.2240 * w_um^2 / t_half_s`

`D = w^2 * k / 4` is valid only when `k` is an inverse time constant (`1/tau`), not `1/t_half`.

## Code Locations (line-resolved)

### Canonical package (`frap2025/`)

- `frap2025/core/frap_fitting.py:183` (`k_s = 1/tau_s`, then `D = w^2*k_s/4`)
- `frap2025/core/frap_fitting.py:262` (single-exp diffusion mapping documentation)
- `frap2025/core/frap_fitting.py:360` (double-exp component diffusion mapping)
- `frap2025/core/frap_fitting.py:617` (Soumpasis `tau_D = w^2/(4D)` mapping)
- `frap2025/core/frap_fitting.py:619` (Soumpasis `0.2240 * w^2/t_half` relation)
- `frap2025/core/frap_fitting.py:775` (reaction-diffusion effective `D = w^2*k_D/4`)

### Legacy root modules still present

- `frap_core.py:162`
- `frap_core.py:2105`
- `frap_bootstrap.py:91`
- `frap_comparison_v2.py:132`
- `frap_comparison_v2.py:141`
- `frap_robust_bayesian.py:796`
- `frap2025/cli.py:125`
- `streamlit_frap.py:140`
- `streamlit_frap_final_clean.py:222`
- `streamlit_frap_final_clean.py:1459`
- `streamlit_frap_final_restored.py:222`
- `streamlit_frap_final_restored.py:1881`
- `streamlit_frap_final_restored.py:2029`
- `frap2025/ui/legacy_app.py:2156`
- `frap2025/ui/legacy_app.py:2197`
- `frap2025/ui/legacy_app.py:2368`
- `frap2025/ui/legacy_app.py:2392`

Each active diffusion computation site now includes an adjacent convention comment clarifying that the fitted rate is `k = 1/tau`.

## Verification Test

Implemented in `tests/test_physics.py`:

- `verify_diffusion_formula()`
- single exponential recovery check
- Soumpasis recovery check
- `tau_s` <-> `t_half_s` consistency check

Run result:

```bash
python -m pytest tests/test_physics.py -v
# 6 passed
```

## Literature References

- Axelrod et al. (1976), *Biophysical Journal* 16:1055-1069. DOI: 10.1016/S0006-3495(76)85755-4
- Soumpasis (1983), *Biophysical Journal* 41:95-97. DOI: 10.1016/S0006-3495(83)84410-5
- Sprague et al. (2004), *Biophysical Journal* 86:3473-3495. DOI: 10.1529/biophysj.103.026765
