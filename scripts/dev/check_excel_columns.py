"""
Quick diagnostic script to check Excel file column names
"""
import pandas as pd
import sys

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    print("Usage: python check_excel_columns.py <path_to_excel_file>")
    sys.exit(1)

try:
    # Try reading the Excel file
    df = pd.read_excel(file_path, engine='xlrd')
    
    print(f"\n✅ Successfully read: {file_path}")
    print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Column names found:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. '{col}'")
    
    print(f"\n👁️ First few rows:")
    print(df.head())
    
except Exception as e:
    print(f"\n❌ Error reading file: {e}")
    print(f"\nTrying to get any info possible...")
    try:
        import xlrd
        wb = xlrd.open_workbook(file_path)
        sheet = wb.sheet_by_index(0)
        print(f"\nSheet name: {sheet.name}")
        print(f"Rows: {sheet.nrows}, Cols: {sheet.ncols}")
        if sheet.nrows > 0:
            print("\nFirst row (possible headers):")
            print([sheet.cell_value(0, col) for col in range(sheet.ncols)])
    except Exception as e2:
        print(f"Also failed: {e2}")
