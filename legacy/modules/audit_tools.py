import importlib.util
import sys
import subprocess
import shutil

def check_package(name, import_name=None):
    if not import_name: import_name = name
    spec = importlib.util.find_spec(import_name)
    return "INSTALLED" if spec else "MISSING"

def check_binary(name):
    return "INSTALLED" if shutil.which(name) else "MISSING (Required for BIOPOINT)"

print(f"{'TOOL / METHOD':<35} | {'STATUS':<12} | {'ACTION'}")
print("-" * 75)

# 1. FlickerPrint (Membrane Tension)
status = check_package("flickerprint")
print(f"{'FlickerPrint':<35} | {status:<12} | {'Add Module A' if status=='MISSING' else 'Skip'}")

# 2. Hi-D (Dense Diffusion Mapping)
status = check_package("hidpy")
print(f"{'Hi-D (hidpy)':<35} | {status:<12} | {'Add Module B' if status=='MISSING' else 'Skip'}")

# 3. SPTnet (Deep Learning Tracking)
# Checks for torch and a heuristic project structure
status = "INSTALLED" if check_package("torch") == "INSTALLED" and check_package("sptnet") == "INSTALLED" else "MISSING"
print(f"{'SPTnet':<35} | {status:<12} | {'Add Module C' if status=='MISSING' else 'Skip'}")

# 4. BIOPOINT (Nuclear Mechanics)
lammps_status = check_binary("lmp_serial")
print(f"{'BIOPOINT (LAMMPS dependency)':<35} | {lammps_status:<12} | {'Add Module D' if lammps_status=='MISSING' else 'Skip'}")

# 5. FIDES (Force Inference)
status = check_package("fides") 
print(f"{'FIDES (Force Inference)':<35} | {status:<12} | {'Add Module E' if status=='MISSING' else 'Skip'}")
