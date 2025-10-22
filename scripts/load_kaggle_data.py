#!/usr/bin/env python3
"""
Load and prepare Kaggle datasets
"""

import pandas as pd
from pathlib import Path

def load_all_datasets():
    """Load and combine all Kaggle datasets"""
    data_dir = Path("data")
    
    # Dataset 1: disease-symptom-description
    df1 = load_symptom_description_dataset(data_dir)
    
    # Dataset 2: Training.csv format
    df2 = load_training_dataset(data_dir)
    
    # Combine
    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.drop_duplicates()
    
    print(f"Total samples: {len(combined)}")
    print(f"Unique diseases: {combined['disease'].nunique()}")
    
    return combined

def load_symptom_description_dataset(data_dir):
    """Load symptom description format"""
    try:
        df = pd.read_csv(data_dir / "dataset.csv")
        # Usually has: Disease, Symptom_1, Symptom_2, etc.
        
        symptom_cols = [col for col in df.columns if 'Symptom' in col]
        
        df['symptoms'] = df[symptom_cols].apply(
            lambda x: ' '.join(x.dropna().astype(str)), 
            axis=1
        )
        df['disease'] = df['Disease']
        
        return df[['symptoms', 'disease']]
    except:
        return pd.DataFrame(columns=['symptoms', 'disease'])

def load_training_dataset(data_dir):
    """Load Training.csv format with prognosis"""
    try:
        df = pd.read_csv(data_dir / "Training.csv")
        
        # Columns are symptoms (0 or 1) and prognosis
        symptom_cols = [col for col in df.columns if col != 'prognosis']
        
        # Get symptom names where value is 1
        df['symptoms'] = df[symptom_cols].apply(
            lambda row: ' '.join([
                col.replace('_', ' ') 
                for col in symptom_cols 
                if row[col] == 1
            ]),
            axis=1
        )
        df['disease'] = df['prognosis']
        
        return df[['symptoms', 'disease']]
    except:
        return pd.DataFrame(columns=['symptoms', 'disease'])

if __name__ == "__main__":
    df = load_all_datasets()
    df.to_csv("data/combined_dataset.csv", index=False)
    print("\nSaved to: data/combined_dataset.csv")