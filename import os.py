import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

# --- Configuration ---
# Update this path to point to your actual data folder
DATA_ROOT = "D:\\PARPFRAP\\265_All" 
OUTPUT_DIR = "D:\\PARPFRAP\\analyzed"

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

def load_frap_data(root_dir, config):
    """
    Crawls the directory and groups data based on the config keywords.
    """
    compiled_data = []
    
    # Try to import the repo's loader
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from frap_data_loader import load_frap_file
        print("Successfully imported 'frap_data_loader'.")
    except ImportError:
        print("Warning: Could not import 'frap_data_loader'. Using fallback JSON/CSV parser.")
        load_frap_file = None

    print(f"\nScanning {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        # Determine which group this folder belongs to
        folder_name = os.path.basename(root)
        matched_group = None
        
        for group_name, keywords in config.items():
            if any(k.lower() in folder_name.lower() for k in keywords):
                # Priority check: ensure we don't accidentally match "PARPi" when looking for "PARP"
                # For this specific logic, exact keyword matching usually works best
                matched_group = group_name
                break
        
        if not matched_group:
            continue

        # Look for valid data files
        data_files = [f for f in files if f.endswith(('.json', '.csv', '.xlsx')) and not f.startswith('~$')]
        
        for f in data_files:
            file_path = os.path.join(root, f)
            results = {}
            
            # 1. Try loading using the repo's tool (best for calculated params)
            if load_frap_file:
                try:
                    obj = load_frap_file(file_path)
                    # Handle different return types (object vs dict)
                    if hasattr(obj, 'results'): results = obj.results
                    elif isinstance(obj, dict): results = obj
                    elif hasattr(obj, '__dict__'): results = obj.__dict__
                except Exception:
                    pass

            # 2. Fallback: Try reading JSON/CSV directly if keys missing
            if not results:
                try:
                    if f.endswith('.json'):
                        import json
                        with open(file_path, 'r') as jf: results = json.load(jf)
                    elif f.endswith('.csv'):
                        df_temp = pd.read_csv(file_path)
                        # Assume simple 1-row CSV with headers
                        if not df_temp.empty: results = df_temp.iloc[0].to_dict()
                except Exception:
                    continue

            # Extract Metrics
            row = {"Group": matched_group, "Filename": f}
            has_data = False
            for display_name, keys in METRICS.items():
                val = find_metric_value(results, keys)
                row[display_name] = val
                if not np.isnan(val): has_data = True
            
            if has_data:
                compiled_data.append(row)

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
    ax = sns.boxplot(x="Group", y=metric, data=df, showfliers=False, 
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

# --- Main Execution Flow ---

def run_split_analysis():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Run Analysis for MUTANTS
    print("\n" + "="*40)
    print("STARTING MUTANT DATASET ANALYSIS")
    print("="*40)
    df_mutants = load_frap_data(DATA_ROOT, MUTANT_CONFIG)
    
    if not df_mutants.empty:
        mutant_out_dir = os.path.join(OUTPUT_DIR, "Mutants")
        os.makedirs(mutant_out_dir, exist_ok=True)
        
        df_mutants.to_csv(os.path.join(mutant_out_dir, "mutant_raw_data.csv"), index=False)
        
        stats_file = open(os.path.join(mutant_out_dir, "mutant_statistics.txt"), "w")
        for metric in METRICS.keys():
            # Check if metric has enough data
            if df_mutants[metric].count() > 1:
                # Plot
                generate_plots(df_mutants, metric, 
                               os.path.join(mutant_out_dir, f"plot_{metric.split()[0]}.png"), 
                               "Mutant Comparison")
                # Stats
                stats_res = perform_statistical_analysis(df_mutants, metric)
                stats_file.write(stats_res + "\n\n" + "-"*30 + "\n\n")
        stats_file.close()
        print(f"Mutant analysis complete. Results in {mutant_out_dir}")
    else:
        print("No matching Mutant data found.")

    # 2. Run Analysis for DRUGS
    print("\n" + "="*40)
    print("STARTING DRUG DATASET ANALYSIS")
    print("="*40)
    df_drugs = load_frap_data(DATA_ROOT, DRUG_CONFIG)
    
    if not df_drugs.empty:
        drug_out_dir = os.path.join(OUTPUT_DIR, "Drugs")
        os.makedirs(drug_out_dir, exist_ok=True)
        
        df_drugs.to_csv(os.path.join(drug_out_dir, "drug_raw_data.csv"), index=False)
        
        stats_file = open(os.path.join(drug_out_dir, "drug_statistics.txt"), "w")
        for metric in METRICS.keys():
            if df_drugs[metric].count() > 1:
                # Plot
                generate_plots(df_drugs, metric, 
                               os.path.join(drug_out_dir, f"plot_{metric.split()[0]}.png"), 
                               "Drug Treatment Comparison")
                # Stats
                stats_res = perform_statistical_analysis(df_drugs, metric)
                stats_file.write(stats_res + "\n\n" + "-"*30 + "\n\n")
        stats_file.close()
        print(f"Drug analysis complete. Results in {drug_out_dir}")
    else:
        print("No matching Drug data found.")

if __name__ == "__main__":
    run_split_analysis()