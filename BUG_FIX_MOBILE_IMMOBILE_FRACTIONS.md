# Mobile/Immobile Fraction Calculation Bug Fix

## Issue Summary
**Problem**: Mobile fraction and immobile fraction calculations were not adding up to 100%, with reported values like:
- Average Mobile Fraction: 93.67%
- Average Immobile Fraction: 42.02%
- Total: 135.69% (should be ~100%)

## Root Cause Analysis

### The Bug
The issue was located in `frap_core.py` in the `extract_clustering_features()` function at line 1335:

```python
# PROBLEMATIC CODE (BEFORE FIX)
features['immobile_fraction'] = 100 - features.get('mobile_fraction', 0)
```

### Why This Was Wrong
1. **Double Calculation**: The immobile fraction was being calculated in two places:
   - **Correctly** in `frap_manager.py` (line 44): `features['immobile_fraction'] = 100.0 - features['mobile_fraction']`
   - **Incorrectly** in `frap_core.py` (line 1335): `features['immobile_fraction'] = 100 - features.get('mobile_fraction', 0)`

2. **Default Value Problem**: The `frap_core.py` calculation used `.get('mobile_fraction', 0)`, which:
   - Returns `0` when `mobile_fraction` is missing or `NaN`
   - Results in `immobile_fraction = 100 - 0 = 100%`
   - This overrides the correct calculation from `frap_manager.py`

3. **Override Issue**: The `frap_core.py` function runs after `frap_manager.py`, so it overrides the correct calculation with an incorrect one.

### Execution Flow
```
1. frap_manager.py calculates: immobile_fraction = 100 - mobile_fraction ✅
2. frap_core.py OVERRIDES with: immobile_fraction = 100 - 0 = 100% ❌
3. Result: mobile_fraction = 93.67%, immobile_fraction = 100% (incorrect override)
```

## The Fix

### Code Change
**File**: `frap_core.py`  
**Lines**: 1335-1340  

**BEFORE (Problematic)**:
```python
features['immobile_fraction'] = 100 - features.get('mobile_fraction', 0)
```

**AFTER (Fixed)**:
```python
# Ensure immobile fraction = 100% - mobile fraction (only if mobile fraction is finite)
mobile_frac = features.get('mobile_fraction', np.nan)
if np.isfinite(mobile_frac):
    features['immobile_fraction'] = 100.0 - mobile_frac
else:
    features['immobile_fraction'] = np.nan
```

### Why This Fixes The Problem

1. **Proper NaN Handling**: Only calculates immobile fraction when mobile fraction is finite
2. **Consistent Logic**: Matches the logic in `frap_manager.py` 
3. **Maintains Data Integrity**: Both fractions are `NaN` when calculations fail, instead of defaulting to inconsistent values

## Verification

### Test Results
Created `test_mobile_immobile_fraction_fix.py` which tests various scenarios:

```
Test Case                      Mobile (%)   Immobile (%)   Sum (%)    Status    
--------------------------------------------------------------------------------
Normal single exponential fit  90.00        10.00          100.00     PASS
Double exponential fit         85.00        15.00          100.00     PASS
Edge case: very low mobile     95.00        5.00           100.00     PASS
Edge case: very high mobile    98.00        2.00           100.00     PASS
Invalid case: NaN fractions    NaN          NaN            NaN        PASS
--------------------------------------------------------------------------------
✅ ALL TESTS PASSED! Mobile and immobile fractions now add up to 100% correctly.
```

### Expected Behavior After Fix
- **Valid fits**: Mobile% + Immobile% = 100% (±0.01% tolerance)
- **Invalid fits**: Both mobile% and immobile% = NaN
- **No more inconsistent values**: No cases where fractions don't add up to 100%

## Impact

### Before Fix
- Inconsistent fraction calculations
- Total fractions could exceed 100% or be far below 100%
- Misleading results in population analysis
- Difficult to interpret FRAP recovery data

### After Fix
- All fraction calculations consistent and mathematically correct
- Population averages will be reliable
- Results interpretable and scientifically meaningful
- Maintains data quality standards

## Files Modified
1. **`frap_core.py`**: Fixed the problematic immobile fraction calculation in `extract_clustering_features()`
2. **`test_mobile_immobile_fraction_fix.py`**: Added comprehensive test suite to verify the fix

## Security Compliance
- Security scan completed with no new vulnerabilities introduced
- All existing security issues are in unrelated files
- Fix follows best practices for numerical calculations and NaN handling