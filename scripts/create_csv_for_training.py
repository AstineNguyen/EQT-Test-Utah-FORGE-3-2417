#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create CSV file for EQTransformer training
"""

import pandas as pd
import numpy as np
import h5py
import os

def create_training_csv():
    """Create CSV file for training"""
    print("=== CREATING TRAINING CSV ===")
    
    # Read the HDF5 file to get trace names
    with h5py.File('training_data/forge/train.hdf5', 'r') as f:
        n_samples = f['X'].shape[0]
    
    # Create trace names
    trace_names = [f"trace_{i:06d}" for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame({'trace_name': trace_names})
    
    # Save CSV
    csv_path = 'training_data/forge/train.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"OK: Created CSV with {len(trace_names)} traces")
    print(f"Saved to: {csv_path}")
    
    return csv_path

def create_test_csv():
    """Create CSV file for testing"""
    print("\n=== CREATING TEST CSV ===")
    
    # Read the HDF5 file to get trace names
    with h5py.File('training_data/forge/test.hdf5', 'r') as f:
        n_samples = f['X'].shape[0]
    
    # Create trace names
    trace_names = [f"test_trace_{i:06d}" for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame({'trace_name': trace_names})
    
    # Save CSV
    csv_path = 'training_data/forge/test.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"OK: Created test CSV with {len(trace_names)} traces")
    print(f"Saved to: {csv_path}")
    
    return csv_path

def main():
    """Main function"""
    print("=== CREATING CSV FILES FOR TRAINING ===")
    
    # Create training CSV
    train_csv = create_training_csv()
    
    # Create test CSV
    test_csv = create_test_csv()
    
    print("\n=== COMPLETED ===")
    print(f"Training CSV: {train_csv}")
    print(f"Test CSV: {test_csv}")

if __name__ == "__main__":
    main()
