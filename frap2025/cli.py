import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import zipfile
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

LOGGER = logging.getLogger(__name__)

# Keep legacy module imports working after root cleanup.
LEGACY_MODULE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legacy", "modules")
if os.path.isdir(LEGACY_MODULE_DIR) and LEGACY_MODULE_DIR not in sys.path:
    sys.path.insert(0, LEGACY_MODULE_DIR)

_LOADER_CACHE = None
_LOADER_RESOLVED = False

# --- Configuration ---
# Update this path to point to your actual data folder
DATA_ROOT = "D:/PARPFRAP/aLL/LaserIR.zip"
OUTPUT_DIR = "D:/PARPFRAP/analyzed"

# Keywords to identify and group folders
# Format: "Display Name": ["keyword1", "keyword2"]
MUTANT_CONFIG = {
    "WT (Control)": ["WT", "Wild Type"],
    "E558A": ["E558A"],
    "H428A": ["H428A"],
    "RAYF (R153A+Y201F)": ["RAYF", "R153A+Y201F", "R153A"]
}

DRUG_CONFIG = {
    "Control (DMSO)": ["ctr", "Control", "DMSO"],
    "Nimbolide": ["nimbolide"],
    "PARPi (AZD/Olaparib)": ["PARPi", "AZD", "5305", "Olaparib"],
    "PARGi": ["PARGi"],
    "DBeQ": ["DBeQ"]
}

METRICS = {
    "Mobile Fraction": ["mobile_fraction", "mobile", "Mobile Fraction"],
    "Diffusion Coefficient (D)": ["diffusion_coeff", "D_eff", "D", "Diffusion Coefficient"],
    "Residency Time (s)": ["residency_time", "tau", "residency", "Residency Time"]
}

# Raw analysis settings
RAW_ANALYSIS_ENABLED = True
PIXEL_SIZE_UM = 0.12
BLEACH_RADIUS_UM_265 = 6 * PIXEL_SIZE_UM
BLEACH_RADIUS_UM_LASER_IR = 12 * PIXEL_SIZE_UM
LASER_IR_KEYWORDS = ["laser", "ir", "infrared"]

# --- Helper Functions ---

def find_metric_value(result_dict, metric_keys):
    """Scans the result dictionary for any of the possible keys for a metric."""
    for key in metric_keys:
        if key in result_dict and result_dict[key] is not None:
            try:
                return float(result_dict[key])
            except (ValueError, TypeError):
                continue
    return np.nan

def _get_loader():
    global _LOADER_CACHE
    global _LOADER_RESOLVED

    if _LOADER_RESOLVED:
        return _LOADER_CACHE

    _LOADER_RESOLVED = True
    try:
        if LEGACY_MODULE_DIR not in sys.path:
            sys.path.append(LEGACY_MODULE_DIR)
        import frap_data_loader as _loader_module

        load_frap_file = getattr(_loader_module, "load_frap_file", None)
        if callable(load_frap_file):
            _LOADER_CACHE = load_frap_file
            LOGGER.info("Using frap_data_loader.load_frap_file")
            return _LOADER_CACHE

        LOGGER.info("frap_data_loader found but no load_frap_file; using JSON/CSV fallback.")
    except Exception as exc:
        LOGGER.info("Could not import frap_data_loader (%s); using JSON/CSV fallback.", exc)

    _LOADER_CACHE = None
    return None

def _iter_data_files(files):
    return [
        f for f in files
        if f.lower().endswith(('.json', '.csv', '.xls', '.xlsx'))
        and not f.startswith('~$')
        and not f.startswith('.')
    ]

def _resolve_bleach_radius_um(group_name: str) -> float:
    name = (group_name or "").lower()
    if any(k in name for k in LASER_IR_KEYWORDS):
        return BLEACH_RADIUS_UM_LASER_IR
    return BLEACH_RADIUS_UM_265

def _analyze_raw_metrics(file_path: str, bleach_radius_um: float) -> dict:
    try:
        from frap_input_handler import FRAPInputHandler
        from frap_core import FRAPAnalysisCore
    except Exception:
        return {}

    try:
        curve = FRAPInputHandler.load_file(file_path)
        bleach_idx = FRAPInputHandler.detect_bleach_frame(curve)
        curve = FRAPInputHandler.double_normalization(curve, bleach_idx)
        curve = FRAPInputHandler.time_zero_correction(curve, bleach_idx)

        time_data = curve.time
        intensity_data = curve.normalized_intensity

        if time_data is None or intensity_data is None:
            return {}

        if len(time_data) > 300:
            idx = np.linspace(0, len(time_data) - 1, 300).astype(int)
            time_data = time_data[idx]
            intensity_data = intensity_data[idx]
        try:
            fit = FRAPAnalysisCore.fit_reaction_diffusion(time_data, intensity_data)
        except Exception:
            fit = None

        if fit and fit.get('success', False):
            params = fit.get('params', [])
            if len(params) >= 5:
                A_diff, k_diff, A_bind, k_off, C = params[:5]
                total_A = float(A_diff) + float(A_bind)
                endpoint = total_A + float(C)
                # k_diff is the reaction-diffusion exponential rate (k_diff = 1/tau_diff), not 1/t_half.
                diffusion_coefficient = (bleach_radius_um ** 2) * float(k_diff) / 4.0 if k_diff > 0 else np.nan
                residence_time = 1.0 / float(k_off) if k_off > 0 else np.nan
                return {
                    'mobile_fraction': endpoint * 100,
                    'diffusion_coeff': diffusion_coefficient,
                    'residency_time': residence_time,
                }

        fit_single = FRAPAnalysisCore.fit_single_exponential(time_data, intensity_data)
        if fit_single and fit_single.get('success', False):
            params = fit_single.get('params', [])
            if len(params) >= 3:
                A, _, C = params[:3]
                endpoint = float(A) + float(C)
                return {
                    'mobile_fraction': endpoint * 100,
                    'diffusion_coeff': np.nan,
                    'residency_time': np.nan,
                }
    except Exception:
        return {}

    return {}

def _load_results_from_file(file_path, filename, load_frap_file):
    results = {}

    if load_frap_file:
        try:
            obj = load_frap_file(file_path)
            if hasattr(obj, 'results'):
                results = obj.results
            elif isinstance(obj, dict):
                results = obj
            elif hasattr(obj, '__dict__'):
                results = obj.__dict__
        except Exception:
            pass

    if not results:
        try:
            if filename.lower().endswith('.json'):
                import json
                with open(file_path, 'r') as jf:
                    results = json.load(jf)
            elif filename.lower().endswith('.csv'):
                df_temp = pd.read_csv(file_path)
                if not df_temp.empty:
                    results = df_temp.iloc[0].to_dict()
        except Exception:
            return {}

    return results

def _build_result_row(file_path, filename, group_name, bleach_radius_um, load_frap_file):
    results = _load_results_from_file(file_path, filename, load_frap_file)
    if not results and RAW_ANALYSIS_ENABLED:
        results = _analyze_raw_metrics(file_path, bleach_radius_um)
    if not results:
        return None

    row = {"Group": group_name, "Filename": filename}
    has_data = False
    for display_name, keys in METRICS.items():
        val = find_metric_value(results, keys)
        row[display_name] = val
        if not np.isnan(val):
            has_data = True
    return row if has_data else None


def _process_file_task(task):
    root_dir, group_name, filename, bleach_radius_um = task
    file_path = os.path.join(root_dir, filename)
    load_frap_file = _get_loader()
    return _build_result_row(file_path, filename, group_name, bleach_radius_um, load_frap_file)


def _collect_rows(root_dir, group_name, files, load_frap_file, compiled_data, bleach_radius_um, parallel=1):
    valid_files = _iter_data_files(files)
    if parallel > 1 and valid_files:
        tasks = [(root_dir, group_name, f, bleach_radius_um) for f in valid_files]
        with mp.Pool(processes=parallel) as pool:
            for row in pool.map(_process_file_task, tasks):
                if row is not None:
                    compiled_data.append(row)
        return

    for f in valid_files:
        file_path = os.path.join(root_dir, f)
        row = _build_result_row(file_path, f, group_name, bleach_radius_um, load_frap_file)
        if row is not None:
            compiled_data.append(row)

def load_frap_data_from_zip(zip_path, parallel=1):
    compiled_data = []
    load_frap_file = _get_loader()

    print(f"\nScanning ZIP {zip_path}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        for root, dirs, files in os.walk(temp_dir):
            if root == temp_dir:
                continue

            folder_name = os.path.basename(root)
            if folder_name.startswith('.') or folder_name.startswith('__'):
                continue

            if not _iter_data_files(files):
                continue

            bleach_radius_um = _resolve_bleach_radius_um(folder_name)
            _collect_rows(
                root,
                folder_name,
                files,
                load_frap_file,
                compiled_data,
                bleach_radius_um,
                parallel=parallel,
            )

    return pd.DataFrame(compiled_data)

def load_frap_data(root_dir, config, parallel=1):
    """
    Crawls the directory and groups data based on the config keywords.
    If a ZIP file is provided, groups are derived from folder names in the ZIP.
    """
    if os.path.isfile(root_dir) and root_dir.lower().endswith('.zip'):
        return load_frap_data_from_zip(root_dir, parallel=parallel)

    compiled_data = []
    load_frap_file = _get_loader()

    print(f"\nScanning {root_dir}...")

    for root, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(root)
        matched_group = None

        for group_name, keywords in config.items():
            if any(k.lower() in folder_name.lower() for k in keywords):
                matched_group = group_name
                break

        if not matched_group:
            continue

        bleach_radius_um = _resolve_bleach_radius_um(folder_name)
        _collect_rows(
            root,
            matched_group,
            files,
            load_frap_file,
            compiled_data,
            bleach_radius_um,
            parallel=parallel,
        )

    return pd.DataFrame(compiled_data)

def perform_statistical_analysis(df, metric, group_col="Group"):
    """
    Performs Kruskal-Wallis (non-parametric ANOVA) followed by Dunn's post-hoc test.
    This is standard for biological datasets with potentially unequal variances/sizes.
    """
    groups = df[group_col].unique()
    if len(groups) < 2:
        return "Not enough groups for statistics."

    group_data = [df[df[group_col] == g][metric].dropna().values for g in groups]
    
    # 1. Kruskal-Wallis Test
    try:
        h_stat, p_val = stats.kruskal(*group_data)
    except ValueError:
        return "Insufficient data for statistics."

    report = f"--- {metric} ---\n"
    report += f"Kruskal-Wallis H-test: H={h_stat:.3f}, p={p_val:.3e}\n"
    
    if p_val < 0.05:
        report += "Significant difference detected (p < 0.05). Running Dunn's post-hoc test...\n\n"
        # 2. Dunn's Post-hoc Test
        try:
            dunn_results = sp.posthoc_dunn(df, val_col=metric, group_col=group_col, p_adjust='bonferroni')
            report += dunn_results.to_string()
        except Exception as e:
            report += f"Could not run post-hoc test: {e}"
    else:
        report += "No significant difference detected among groups.\n"
        
    return report

def generate_plots(df, metric, output_path, title_prefix):
    """Generates a publication-quality boxplot with strip overlay."""
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Create Boxplot
    sns.boxplot(x="Group", y=metric, data=df, showfliers=False,
                palette="viridis", boxprops=dict(alpha=.6))
    
    # Overlay individual points (Strip plot)
    sns.stripplot(x="Group", y=metric, data=df, 
                  color=".2", alpha=0.6, jitter=True, size=5)
    
    plt.title(f"{title_prefix}: {metric}", fontsize=15, fontweight='bold')
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()

def _write_group_analysis(df, output_subdir, title_prefix, csv_name, out_root=OUTPUT_DIR):
    if df.empty:
        print(f"No matching {title_prefix} data found.")
        return

    out_dir = os.path.join(out_root, output_subdir) if output_subdir else out_root
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, csv_name), index=False)

    stats_file = open(os.path.join(out_dir, "statistics.txt"), "w")
    for metric in METRICS.keys():
        if df[metric].count() > 1:
            generate_plots(
                df,
                metric,
                os.path.join(out_dir, f"plot_{metric.split()[0]}.png"),
                title_prefix,
            )
            stats_res = perform_statistical_analysis(df, metric)
            stats_file.write(stats_res + "\n\n" + "-" * 30 + "\n\n")
    stats_file.close()
    print(f"{title_prefix} analysis complete. Results in {out_dir}")

# --- Main Execution Flow ---

def run_split_analysis(
    *,
    data_root=DATA_ROOT,
    output_dir=OUTPUT_DIR,
    mutant_config=None,
    drug_config=None,
    parallel=1,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mutant_config = mutant_config or MUTANT_CONFIG
    drug_config = drug_config or DRUG_CONFIG

    if os.path.isfile(data_root) and data_root.lower().endswith('.zip'):
        print("\n" + "="*40)
        print("STARTING ZIP DATASET ANALYSIS")
        print("="*40)
        df_all = load_frap_data_from_zip(data_root, parallel=parallel)
        _write_group_analysis(df_all, "ZipGroups", "ZIP Group Comparison", "zip_raw_data.csv", out_root=output_dir)
        return

    # 1. Run Analysis for MUTANTS
    print("\n" + "="*40)
    print("STARTING MUTANT DATASET ANALYSIS")
    print("="*40)
    df_mutants = load_frap_data(data_root, mutant_config, parallel=parallel)
    
    _write_group_analysis(df_mutants, "Mutants", "Mutant Comparison", "mutant_raw_data.csv", out_root=output_dir)

    # 2. Run Analysis for DRUGS
    print("\n" + "="*40)
    print("STARTING DRUG DATASET ANALYSIS")
    print("="*40)
    df_drugs = load_frap_data(data_root, drug_config, parallel=parallel)
    
    _write_group_analysis(df_drugs, "Drugs", "Drug Treatment Comparison", "drug_raw_data.csv", out_root=output_dir)

def _load_cli_config(config_path):
    if not config_path:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "YAML config support requires PyYAML. Install with `pip install pyyaml`."
        ) from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must define a YAML mapping/dictionary.")
    return payload


def run_validation_suite():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "pytest", "tests/test_physics.py", "-v"]
    result = subprocess.run(cmd, cwd=repo_root, check=False)
    return result.returncode


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="FRAP2025 CLI analysis entrypoint.")
    parser.add_argument("input_path", nargs="?", default=DATA_ROOT, help="Input directory or ZIP path.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file overriding mutant/drug keyword groups and output_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Worker count for batch file processing (uses multiprocessing.Pool).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run synthetic physics validation tests and exit.",
    )
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.validate:
        return run_validation_suite()

    cfg = _load_cli_config(args.config) if args.config else {}
    mutant_cfg = cfg.get("mutant_config", MUTANT_CONFIG)
    drug_cfg = cfg.get("drug_config", DRUG_CONFIG)
    output_dir = cfg.get("output_dir", args.output_dir)

    run_split_analysis(
        data_root=args.input_path,
        output_dir=output_dir,
        mutant_config=mutant_cfg,
        drug_config=drug_cfg,
        parallel=max(1, int(args.parallel)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
