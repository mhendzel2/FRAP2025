# FRAP Models

This document defines the model conventions used in `frap2025/core/frap_fitting.py`.

## Parameter Convention

- Exponential terms use `exp(-t_s / tau_s)` equivalently `exp(-k_s * t_s)` with `k_s = 1/tau_s`.
- Diffusion conversion for exponential timescales:
  - `D_um2_per_s = w_um^2 / (4 * tau_s)`
  - `D_um2_per_s = w_um^2 * k_s / 4` (same equation)
- Half-time conversion:
  - `t_half_s = tau_s * ln(2)`
  - `D_um2_per_s = w_um^2 * ln(2) / (4 * t_half_s)`

## Model 1: Single Exponential

- Equation:
  - `F(t) = baseline + mobile_fraction * (1 - exp(-t/tau_s))`
- Diffusion estimate:
  - `D_um2_per_s = w_um^2 / (4*tau_s)`
- Reference:
  - Axelrod et al. (1976), *Biophys J* 16:1055-1069, DOI: 10.1016/S0006-3495(76)85755-4

## Model 2: Double Exponential

- Equation:
  - `F(t) = baseline + A1*(1-exp(-t/tau1_s)) + A2*(1-exp(-t/tau2_s))`
- Component diffusion estimates:
  - `D1_um2_per_s = w_um^2 / (4*tau1_s)`
  - `D2_um2_per_s = w_um^2 / (4*tau2_s)`
- Mobile fraction:
  - `mobile_fraction = A1 + A2`

## Model 3: Soumpasis Exact Diffusion

- Equation:
  - `F(t) = baseline + M * exp(-tau_D_s/(2t)) * [I0(z) + I1(z)]`
  - with `z = tau_D_s/(2t)`
- Parameter relation:
  - `tau_D_s = w_um^2 / (4*D_um2_per_s)`
  - `D_um2_per_s = w_um^2 / (4*tau_D_s)`
- Half-time approximation:
  - `D_um2_per_s ~= 0.2240 * w_um^2 / t_half_s`
- Reference:
  - Soumpasis (1983), *Biophys J* 41:95-97, DOI: 10.1016/S0006-3495(83)84410-5

## Model 4: Reaction-Diffusion

- Equation:
  - `F(t)=baseline + M*[1 - (k_off/(k_on+k_off))*exp(-(k_on+k_off)t) - (k_on/(k_on+k_off))*exp(-k_D*t)]`
- Effective diffusion mapping (if bleach radius provided):
  - `D_um2_per_s = w_um^2 * k_D_s / 4`
- Reference:
  - Sprague et al. (2004), *Biophys J* 86:3473-3495, DOI: 10.1529/biophysj.103.026765
