#!/usr/bin/env python3
"""
MediMind GPU-Accelerated Training Script
Trains disease prediction models using GPU acceleration with PyTorch and advanced ML
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Utilities
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Robust NLTK setup: auto-download if missing
import nltk
required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for pkg in required_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}') if pkg=='punkt' else nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)


class Config:
    """Training configuration"""
    # Paths
    DATA_DIR = Path("../data")  # Correct relative path to data
    MODEL_DIR = Path("../models")
    
    # Dataset files
    DATASET_FILES = [
        "dataset.csv",
        "Symptom-severity.csv",
        "symptom_Description.csv",
        "symptom_precaution.csv"
    ]
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # GPU settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.0005
    
    # Feature engineering
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 3)
    
    def __init__(self):
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)


class SymptomsDataset(Dataset):
    """PyTorch Dataset for symptoms"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.toarray())
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralClassifier(nn.Module):
    """Neural Network for disease classification"""
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128]):
        super(NeuralClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MediMindTrainerGPU:
    """GPU-Accelerated trainer for disease prediction"""
    
    def __init__(self, config):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.label_encoder = None
        self.models = {}
        
        print(f"🚀 Initializing MediMind GPU Trainer")
        print(f"📊 Device: {self.config.DEVICE}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def preprocess_text(self, text):
       import nltk
       from nltk.stem import WordNetLemmatizer
       from nltk.tokenize import word_tokenize

       # Ensure punkt is available
       try:
           _ = word_tokenize("test")
       except LookupError:
           nltk.download('punkt', quiet=True)

       if pd.isna(text):
           return ""
    
       text = str(text).lower()
       text = re.sub(r'[^a-zA-Z\s]', '', text)
    
       tokens = word_tokenize(text)
       tokens = [
           self.lemmatizer.lemmatize(word) 
           for word in tokens 
           if word not in self.stop_words and len(word) > 2
       ]
    
       return ' '.join(tokens)


    
    def load_kaggle_dataset(self):
        """Load and merge Kaggle datasets"""
        print("\n📂 Loading Kaggle datasets...")

        dfs = []

        for file in self.config.DATASET_FILES:
            file_path = self.config.DATA_DIR / file
            if file_path.exists():
                print(f"  ✓ Loading {file}")
                try:
                    df = pd.read_csv(file_path)
                    processed_df = self.process_dataset(df)
                    if not processed_df.empty:
                        dfs.append(processed_df)
                    else:
                        print(f"    ⚠ {file} has no valid symptom/disease columns.")
                except Exception as e:
                    print(f"  ✗ Error loading {file}: {e}")
            else:
                print(f"  ⚠ File {file} not found, skipping.")

        if not dfs:
            raise ValueError("No valid datasets found!")

        # Combine all datasets
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['symptoms', 'disease'])

        print(f"\n✅ Combined dataset shape: {combined_df.shape}")
        print(f"📊 Unique diseases: {combined_df['disease'].nunique()}")
        print(f"📝 Total samples: {len(combined_df)}")

        return combined_df
    
    def process_dataset(self, df):
        """
        Process individual dataset to standard format.
        Fully plug-and-play: automatically detects symptom & disease columns.
        """
        # First, standardize column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Detect symptom columns
        symptom_cols = [col for col in df.columns if 'symptom' in col or 'symptoms' in col]
        # Detect disease columns
        disease_cols = [col for col in df.columns if 'disease' in col or 'prognosis' in col]

        if not symptom_cols or not disease_cols:
            print("⚠ Warning: No valid symptom/disease columns found, skipping dataset.")
            return pd.DataFrame(columns=['symptoms', 'disease'])

        # Combine multiple symptom columns if present
        if len(symptom_cols) > 1:
            df['symptoms'] = df[symptom_cols].astype(str).agg(' '.join, axis=1)
        else:
            df['symptoms'] = df[symptom_cols[0]].astype(str)

        # Use first detected disease column
        df['disease'] = df[disease_cols[0]].astype(str)

        # Keep only standard columns
        df = df[['symptoms', 'disease']]

        # Drop empty or invalid rows
        df = df.dropna(subset=['symptoms', 'disease'])
        df = df[df['symptoms'].str.strip() != '']
        df = df[df['disease'].str.strip() != '']

        return df
    
    def augment_data(self, df, augmentation_factor=2):
       """Data augmentation with synonym replacement"""
       print(f"\n🔄 Augmenting data (factor: {augmentation_factor})...")
    
       augmented = []
    
       for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
           # Convert original row to dict
           augmented.append(row.to_dict())
        
           # Create variations
           for _ in range(augmentation_factor - 1):
               words = row['symptoms'].split()
               if len(words) > 3:
                   # Shuffle words slightly
                   np.random.shuffle(words)
                   augmented.append({
                       'symptoms': ' '.join(words),
                       'disease': row['disease']
                   })
    
       aug_df = pd.DataFrame(augmented)
       print(f"✅ Augmented dataset: {len(df)} → {len(aug_df)} samples")
    
       return aug_df
     
    def create_features(self, df):
        """Create TF-IDF features"""
        print("\n🔨 Creating features...")
        
        # Preprocess
        df['processed'] = df['symptoms'].apply(self.preprocess_text)
        
        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.MAX_FEATURES,
            ngram_range=self.config.NGRAM_RANGE,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        X = self.vectorizer.fit_transform(df['processed'])
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['disease'])
        
        print(f"✅ Feature matrix: {X.shape}")
        print(f"✅ Classes: {len(self.label_encoder.classes_)}")
        
        return X, y
    
    def train_traditional_models(self, X_train, y_train, X_test, y_test):
        """Train traditional ML models"""
        print("\n🤖 Training Traditional ML Models...")
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                min_samples_split=5,
                n_jobs=-1,
                random_state=self.config.RANDOM_STATE
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                random_state=self.config.RANDOM_STATE
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                device='cpu',
                random_state=self.config.RANDOM_STATE
            ),
            'CatBoost': CatBoostClassifier(
                iterations=200,
                depth=10,
                learning_rate=0.1,
                task_type='GPU' if torch.cuda.is_available() else 'CPU',
                verbose=0,
                random_state=self.config.RANDOM_STATE
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n  Training {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'train_time': train_time
            }
            
            print(f"    ✓ Accuracy: {accuracy:.4f}")
            print(f"    ✓ F1 Score: {f1:.4f}")
            print(f"    ✓ Time: {train_time:.2f}s")
            
            self.models[name] = model
        
        return results
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train neural network on GPU"""
        print("\n🧠 Training Neural Network on GPU...")
        
        # Create datasets
        train_dataset = SymptomsDataset(X_train, y_train)
        test_dataset = SymptomsDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE
        )
        
        # Model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        model = NeuralClassifier(input_dim, num_classes).to(self.config.DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        # Training
        best_accuracy = 0
        history = {'train_loss': [], 'test_acc': []}
        
        for epoch in range(self.config.EPOCHS):
            model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(self.config.DEVICE)
                y_batch = y_batch.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.config.DEVICE)
                    y_batch = y_batch.to(self.config.DEVICE)
                    
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            accuracy = correct / total
            avg_loss = train_loss / len(train_loader)
            
            history['train_loss'].append(avg_loss)
            history['test_acc'].append(accuracy)
            
            scheduler.step(avg_loss)
            
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), self.config.MODEL_DIR / 'neural_model.pth')
        
        self.models['Neural Network'] = model
        
        return {
            'model': model,
            'accuracy': best_accuracy,
            'history': history
        }
    
    def create_ensemble(self):
        """Create ensemble of best models"""
        print("\n🎯 Creating Ensemble Model...")
        
        # Select top models (excluding neural network)
        traditional_models = [
            ('rf', self.models['Random Forest']),
            ('xgb', self.models['XGBoost']),
            ('lgb', self.models['LightGBM']),
            ('cat', self.models['CatBoost'])
        ]
        
        ensemble = VotingClassifier(
            estimators=traditional_models,
            voting='soft'
        )
        
        self.models['Ensemble'] = ensemble
        
        return ensemble
    
    def save_models(self):
        """Save all models"""
        print("\n💾 Saving models...")
        
        # Save vectorizer and encoder
        with open(self.config.MODEL_DIR / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(self.config.MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save traditional models
        for name, model in self.models.items():
            if name != 'Neural Network':
                filename = name.lower().replace(' ', '_') + '.pkl'
                with open(self.config.MODEL_DIR / filename, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  ✓ Saved {name}")
        
        print("✅ All models saved successfully!")
    
    def evaluate_all_models(self, X_test, y_test):
        """Comprehensive evaluation"""
        print("\n📊 Model Evaluation Summary")
        print("="*60)
        
        results = []
        
        for name, model in self.models.items():
            if name == 'Neural Network':
                # Evaluate neural network
                dataset = SymptomsDataset(X_test, y_test)
                loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
                
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for X_batch, y_batch in loader:
                        X_batch = X_batch.to(self.config.DEVICE)
                        y_batch = y_batch.to(self.config.DEVICE)
                        
                        outputs = model(X_batch)
                        _, predicted = torch.max(outputs, 1)
                        total += y_batch.size(0)
                        correct += (predicted == y_batch).sum().item()
                
                accuracy = correct / total
            else:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
            
            results.append({'Model': name, 'Accuracy': accuracy})
            print(f"{name:20s}: {accuracy:.4f}")
        
        print("="*60)
        
        return pd.DataFrame(results)


def main():
    """Main training pipeline"""
    print("="*60)
    print("🏥 MediMind GPU-Accelerated Training Pipeline")
    print("="*60)
    
    # Configuration
    config = Config()
    trainer = MediMindTrainerGPU(config)
    
    # Load data
    df = trainer.load_kaggle_dataset()
    
    # Augment
    df = trainer.augment_data(df, augmentation_factor=2)
    
    # Create features
    X, y = trainer.create_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    
    # Train models
    trad_results = trainer.train_traditional_models(X_train, y_train, X_test, y_test)
    nn_results = trainer.train_neural_network(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    ensemble = trainer.create_ensemble()
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    results_df = trainer.evaluate_all_models(X_test, y_test)
    
    # Save
    trainer.save_models()
    
    print("\n🎉 Training Complete!")
    print("\nBest Models:")
    print(results_df.sort_values('Accuracy', ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
