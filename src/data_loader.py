import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class MedicalDataLoader:
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.disease_encoder = LabelEncoder()
        self.symptom_vocab = {}
        self.disease_info = {}
        
    def load_datasets(self):
        """Load all medical datasets"""
        # Main disease-symptom dataset
        disease_df = pd.read_csv(os.path.join(self.data_path, 'dataset.csv'))
        
        # Additional information datasets
        try:
            description_df = pd.read_csv(os.path.join(self.data_path, 'symptom_Description.csv'))
            precaution_df = pd.read_csv(os.path.join(self.data_path, 'symptom_precaution.csv'))
            severity_df = pd.read_csv(os.path.join(self.data_path, 'Symptom-severity.csv'))
            
            return disease_df, description_df, precaution_df, severity_df
        except:
            print("Additional datasets not found. Using only main dataset.")
            return disease_df, None, None, None
    
    def preprocess_disease_data(self, disease_df):
        """Preprocess the main disease-symptom dataset"""
        print("Preprocessing disease-symptom data...")
        
        # Get all symptom columns (typically Symptom_1 to Symptom_17)
        symptom_cols = [col for col in disease_df.columns if 'Symptom' in col]
        
        # Create symptom vocabulary
        all_symptoms = set()
        for col in symptom_cols:
            symptoms = disease_df[col].dropna().unique()
            all_symptoms.update(symptoms)
        
        all_symptoms = sorted(list(all_symptoms))
        self.symptom_vocab = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
        
        print(f"Total unique symptoms: {len(self.symptom_vocab)}")
        print(f"Total diseases: {disease_df['Disease'].nunique()}")
        
        # Encode diseases
        disease_df['Disease_Encoded'] = self.disease_encoder.fit_transform(disease_df['Disease'])
        
        # Create binary symptom vectors
        symptom_vectors = []
        disease_labels = []
        
        for idx, row in disease_df.iterrows():
            # Create binary vector for symptoms
            symptom_vector = np.zeros(len(self.symptom_vocab))
            
            for col in symptom_cols:
                symptom = row[col]
                if pd.notna(symptom) and symptom in self.symptom_vocab:
                    symptom_vector[self.symptom_vocab[symptom]] = 1
            
            symptom_vectors.append(symptom_vector)
            disease_labels.append(row['Disease_Encoded'])
        
        X = np.array(symptom_vectors)
        y = np.array(disease_labels)
        
        return X, y, disease_df
    
    def load_disease_info(self, description_df, precaution_df):
        """Load disease descriptions and precautions"""
        if description_df is not None:
            for idx, row in description_df.iterrows():
                disease = row['Disease']
                if disease not in self.disease_info:
                    self.disease_info[disease] = {}
                self.disease_info[disease]['description'] = row.get('Description', 'N/A')
        
        if precaution_df is not None:
            for idx, row in precaution_df.iterrows():
                disease = row['Disease']
                if disease not in self.disease_info:
                    self.disease_info[disease] = {}
                
                precautions = []
                for i in range(1, 5):
                    prec_col = f'Precaution_{i}'
                    if prec_col in row and pd.notna(row[prec_col]):
                        precautions.append(row[prec_col])
                
                self.disease_info[disease]['precautions'] = precautions
    
    def save_preprocessed_data(self, X, y, save_path='data/processed/'):
        """Save preprocessed data and encoders"""
        os.makedirs(save_path, exist_ok=True)
        
        np.save(os.path.join(save_path, 'X_data.npy'), X)
        np.save(os.path.join(save_path, 'y_data.npy'), y)
        
        with open(os.path.join(save_path, 'encoders.pkl'), 'wb') as f:
            pickle.dump({
                'disease_encoder': self.disease_encoder,
                'symptom_vocab': self.symptom_vocab,
                'disease_info': self.disease_info
            }, f)
        
        print(f"Preprocessed data saved to {save_path}")

# PyTorch Dataset class
class SymptomDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
