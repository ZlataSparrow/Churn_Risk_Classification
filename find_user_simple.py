#!/usr/bin/env python3
"""
Simple script to find user at index 10 using pandas only
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the data from the parquet file
try:
    # Try to read the parquet file directly
    df = pd.read_parquet("data/cleaned_churn_data.parquet")
    print("✅ Successfully loaded data from parquet file")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
except Exception as e:
    print(f"❌ Error loading parquet file: {e}")
    print("Let's try to find the user data in a different way...")
    
    # Try to find any CSV or other data files
    import os
    data_files = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(('.csv', '.json', '.parquet')):
                data_files.append(os.path.join(root, file))
    
    print(f"Found data files: {data_files}")
    
    if data_files:
        print(f"Trying to read: {data_files[0]}")
        try:
            if data_files[0].endswith('.json'):
                df = pd.read_json(data_files[0])
            elif data_files[0].endswith('.csv'):
                df = pd.read_csv(data_files[0])
            print("✅ Successfully loaded data")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e2:
            print(f"❌ Error loading data file: {e2}")
