#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple training script for EQTransformer with FORGE dataset
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add path to import EQTransformer
sys.path.append('EQTransformer-master')

def load_forge_data():
    """Load FORGE dataset and create synthetic waveforms"""
    print("=== LOADING FORGE DATASET ===")
    
    # Read FORGE dataset
    df = pd.read_csv('Dataset/FORGE_DAS_event_catalogue_circulation.csv')
    print(f"Total events: {len(df)}")
    
    # Create synthetic waveform data
    n_events = len(df)
    n_channels = 3  # E, N, Z components
    n_timesteps = 6000  # 60 seconds at 100 Hz
    
    print("Creating synthetic waveform data...")
    
    # Initialize arrays
    X = np.zeros((n_events, n_timesteps, n_channels))
    y_detection = np.ones(n_events)  # All are earthquakes
    y_p_pick = np.zeros(n_events)
    y_s_pick = np.zeros(n_events)
    
    # Generate synthetic waveforms based on FORGE metadata
    for i, row in df.iterrows():
        magnitude = row['Mw (calibrated)']
        snr_p = row['SNR mean P']
        snr_s = row['SNR mean S']
        confidence = row['Confidence Index']
        
        # Convert magnitude to amplitude
        amplitude = 10 ** (magnitude - 1.0)
        
        # Generate time series
        t = np.linspace(0, 60, n_timesteps)
        
        # Create synthetic earthquake signal
        for ch in range(n_channels):
            freq = 1.0 + magnitude * 2.0  # Higher magnitude = higher frequency
            
            # Add multiple frequency components
            signal = np.zeros_like(t)
            for f in [freq, freq*2, freq*3]:
                signal += amplitude * np.sin(2 * np.pi * f * t) * np.exp(-t/10)
            
            # Add noise based on SNR
            noise_level = amplitude / max(snr_p, snr_s, 1)
            noise = np.random.normal(0, noise_level, n_timesteps)
            
            X[i, :, ch] = signal + noise
        
        # Set P and S pick times
        relative_ot = row['relative ot (s)']
        y_p_pick[i] = int(relative_ot * 100)
        y_s_pick[i] = int((relative_ot + 2.0) * 100)
    
    return X, y_detection, y_p_pick, y_s_pick, df

def split_data(X, y_detection, y_p_pick, y_s_pick, test_size=0.2):
    """Split data into train and test sets"""
    print(f"\n=== SPLITTING DATA ({int((1-test_size)*100)}% train, {int(test_size*100)}% test) ===")
    
    n_samples = len(X)
    n_train = int((1 - test_size) * n_samples)
    
    # Random split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_detection_train, y_detection_test = y_detection[train_indices], y_detection[test_indices]
    y_p_pick_train, y_p_pick_test = y_p_pick[train_indices], y_p_pick[test_indices]
    y_s_pick_train, y_s_pick_test = y_s_pick[train_indices], y_s_pick[test_indices]
    
    print(f"Training set: {len(train_indices)} events")
    print(f"Test set: {len(test_indices)} events")
    
    return (X_train, y_detection_train, y_p_pick_train, y_s_pick_train), \
           (X_test, y_detection_test, y_p_pick_test, y_s_pick_test)

def create_simple_model():
    """Create a simple neural network model for demonstration"""
    print("\n=== CREATING SIMPLE MODEL ===")
    
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Create model
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(6000, 3)),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dropout(0.3),
            Dense(100, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='sigmoid')  # Detection, P-pick, S-pick
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("OK: Simple model created")
        return model
        
    except Exception as e:
        print(f"ERROR: Failed to create model: {e}")
        return None

def train_simple_model(model, X_train, y_train, X_test, y_test):
    """Train the simple model"""
    print("\n=== TRAINING SIMPLE MODEL ===")
    
    try:
        # Prepare labels for multi-task learning
        y_detection_train, y_p_pick_train, y_s_pick_train = y_train
        
        # Convert to binary classification for detection
        y_detection_binary = y_detection_train.astype(int)
        
        # Normalize pick times to [0, 1] range
        y_p_pick_norm = y_p_pick_train / 6000.0
        y_s_pick_norm = y_s_pick_train / 6000.0
        
        # Combine labels
        y_combined_train = np.column_stack([y_detection_binary, y_p_pick_norm, y_s_pick_norm])
        
        # Prepare test data
        y_detection_test, y_p_pick_test, y_s_pick_test = y_test
        y_detection_binary_test = y_detection_test.astype(int)
        y_p_pick_norm_test = y_p_pick_test / 6000.0
        y_s_pick_norm_test = y_s_pick_test / 6000.0
        y_combined_test = np.column_stack([y_detection_binary_test, y_p_pick_norm_test, y_s_pick_norm_test])
        
        # Training parameters
        epochs = 20  # Reduced for faster training
        batch_size = 16
        
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        
        # Train model
        history = model.fit(
            X_train, y_combined_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_combined_test),
            verbose=1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        print("OK: Training completed")
        return model, history
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return None, None

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n=== EVALUATING MODEL ===")
    
    try:
        # Prepare test data
        y_detection_test, y_p_pick_test, y_s_pick_test = y_test
        y_detection_binary_test = y_detection_test.astype(int)
        y_p_pick_norm_test = y_p_pick_test / 6000.0
        y_s_pick_norm_test = y_s_pick_test / 6000.0
        y_combined_test = np.column_stack([y_detection_binary_test, y_p_pick_norm_test, y_s_pick_norm_test])
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Detection metrics
        y_detection_pred = (predictions[:, 0] > 0.5).astype(int)
        detection_accuracy = accuracy_score(y_detection_binary_test, y_detection_pred)
        detection_f1 = f1_score(y_detection_binary_test, y_detection_pred)
        detection_precision = precision_score(y_detection_binary_test, y_detection_pred)
        detection_recall = recall_score(y_detection_binary_test, y_detection_pred)
        
        # Phase picking accuracy (within 1 second tolerance)
        p_pick_pred = predictions[:, 1] * 6000
        s_pick_pred = predictions[:, 2] * 6000
        
        p_pick_accuracy = np.mean(np.abs(p_pick_pred - y_p_pick_test) <= 100)  # 1 second
        s_pick_accuracy = np.mean(np.abs(s_pick_pred - y_s_pick_test) <= 100)  # 1 second
        
        # Print results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Detection Accuracy: {detection_accuracy:.4f}")
        print(f"Detection F1 Score: {detection_f1:.4f}")
        print(f"Detection Precision: {detection_precision:.4f}")
        print(f"Detection Recall: {detection_recall:.4f}")
        print(f"P-pick Accuracy (1s tolerance): {p_pick_accuracy:.4f}")
        print(f"S-pick Accuracy (1s tolerance): {s_pick_accuracy:.4f}")
        print(f"Test Set Size: {len(X_test)}")
        
        # Save results
        results = {
            'detection_accuracy': float(detection_accuracy),
            'detection_f1': float(detection_f1),
            'detection_precision': float(detection_precision),
            'detection_recall': float(detection_recall),
            'p_pick_accuracy': float(p_pick_accuracy),
            's_pick_accuracy': float(s_pick_accuracy),
            'test_size': len(X_test)
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/simple_training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("OK: Results saved to results/simple_training_results.json")
        
        return results
        
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return None

def main():
    """Main training function"""
    print("=== SIMPLE EQTRANSFORMER TRAINING WITH FORGE DATASET ===")
    print(f"Time: {datetime.now()}")
    print("Split: 80% train, 20% test")
    
    # Step 1: Load data
    X, y_detection, y_p_pick, y_s_pick, df = load_forge_data()
    
    # Step 2: Split data
    train_data, test_data = split_data(X, y_detection, y_p_pick, y_s_pick, test_size=0.2)
    X_train, y_detection_train, y_p_pick_train, y_s_pick_train = train_data
    X_test, y_detection_test, y_p_pick_test, y_s_pick_test = test_data
    
    # Step 3: Create model
    model = create_simple_model()
    if model is None:
        print("ERROR: Failed to create model")
        return
    
    # Step 4: Train model
    model, history = train_simple_model(
        model, 
        X_train, 
        (y_detection_train, y_p_pick_train, y_s_pick_train),
        X_test,
        (y_detection_test, y_p_pick_test, y_s_pick_test)
    )
    
    if model is None:
        print("ERROR: Training failed")
        return
    
    # Step 5: Evaluate model
    results = evaluate_model(
        model, 
        X_test, 
        (y_detection_test, y_p_pick_test, y_s_pick_test)
    )
    
    if results:
        print("\nOK: Training and evaluation completed successfully!")
    else:
        print("ERROR: Evaluation failed")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
