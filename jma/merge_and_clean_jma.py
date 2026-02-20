#!/usr/bin/env python3
"""
merge_and_clean_jma.py
======================
Merges all individual JMA weather CSV files from the orig/ subfolder and cleans the data.

Processes:
  1. Loads all jma_*.csv files from orig/ subfolder with cp932 encoding
  2. Extracts weather columns: timestamp, precip, temp, sun, wind
  3. Removes duplicates and sorts chronologically
  4. Interpolates/fills missing values
  5. Outputs cleaned merged CSV to jma_hourly_cleaned_merged.csv

Usage:
  python3 merge_and_clean_jma.py
"""

import pandas as pd
import glob
from datetime import datetime, timedelta
import os

def main():
    print("=" * 70)
    print("MERGING AND CLEANING ALL JMA FILES")
    print("=" * 70)
    
    all_dfs = []
    file_stats = []
    
    # Get original data directory
    orig_dir = os.path.join(os.path.dirname(__file__), 'orig')
    
    # Load all jma_*.csv files from orig/ subfolder with cp932 encoding
    data_files = sorted(glob.glob(os.path.join(orig_dir, 'jma_*.csv')))
    print(f"\n1. Loading {len(data_files)} jma_*.csv files from orig/ subfolder...")
    
    for fname in data_files:
        try:
            df = pd.read_csv(fname, encoding='cp932', skiprows=5, header=None)
            df = df[[0, 1, 4, 7, 10]].copy()
            df.columns = ['timestamp', 'precip', 'temp', 'sun', 'wind']
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            all_dfs.append(df)
            file_stats.append((os.path.basename(fname), len(df)))
            print(f"   ✓ {os.path.basename(fname):35s}  {len(df):5d} rows")
        except Exception as e:
            print(f"   ✗ {os.path.basename(fname):35s}  ERROR: {str(e)[:40]}")
    
    if not all_dfs:
        print("\n✗ No files loaded. Check orig/ folder exists and contains jma_*.csv files.")
        return
    
    # Merge all
    print(f"\n2. Merging {len(all_dfs)} files...")
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"   Before dedup: {len(merged)} rows")
    
    # Sort and remove duplicates
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    merged = merged.drop_duplicates(subset=['timestamp'], keep='first')
    print(f"   After dedup:  {len(merged)} rows")
    
    # Data cleaning
    print(f"\n3. Data cleaning...")
    print(f"   Missing values BEFORE:")
    missing_before = {}
    for col in merged.columns:
        n_missing = merged[col].isna().sum()
        missing_before[col] = n_missing
        if n_missing > 0:
            print(f"     {col:15s}  {n_missing:6d} ({100*n_missing/len(merged):5.2f}%)")
    
    # Fill minor gaps with interpolation (for sparse missing values)
    for col in ['precip', 'temp', 'sun', 'wind']:
        if merged[col].isna().sum() > 0:
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
    
    # Fill any remaining with forward fill
    merged = merged.ffill()
    merged = merged.bfill()
    
    print(f"   After filling: {merged.isna().sum().sum()} total missing values")
    
    # Numeric validation
    print(f"\n4. Numeric range checks:")
    for col in ['precip', 'temp', 'sun', 'wind']:
        print(f"   {col:15s}  min={merged[col].min():8.1f}  max={merged[col].max():8.1f}  mean={merged[col].mean():8.1f}")
    
    # Date coverage
    dates_covered = merged['timestamp'].dt.date.unique()
    dates_covered = sorted(dates_covered)
    
    print(f"\n5. Date coverage:")
    print(f"   First date:  {dates_covered[0]}")
    print(f"   Last date:   {dates_covered[-1]}")
    print(f"   Unique dates: {len(dates_covered)}")
    
    # Check for target range (camera data range)
    expected_start = datetime.strptime('2024-12-20', '%Y-%m-%d').date()
    expected_end = datetime.strptime('2026-02-18', '%Y-%m-%d').date()
    
    current = expected_start
    missing = []
    while current <= expected_end:
        if current not in dates_covered:
            missing.append(current)
        current += timedelta(days=1)
    
    print(f"   Target range (camera data): {expected_start} to {expected_end}")
    if missing:
        print(f"   ⚠ Missing {len(missing)} days in target range")
    else:
        print(f"   ✓ Complete coverage of camera date range!")
    
    # Save cleaned version with date range in filename
    first_date = dates_covered[0].strftime('%Y-%m-%d')
    last_date = dates_covered[-1].strftime('%Y-%m-%d')
    output_file = os.path.join(os.path.dirname(__file__), f'jma_hourly_cleaned_merged_{first_date}_{last_date}.csv')
    merged.to_csv(output_file, index=False)
    print(f"\n✓ Saved cleaned merged data to: {os.path.basename(output_file)}")
    print(f"  Total rows: {len(merged)}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    print(f"\n{'='*70}")
    print("MERGE AND CLEAN COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
