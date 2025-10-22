import numpy as np
import torch
from torch.utils.data import Dataset

class AugmentedSymptomDataset(Dataset):
    """Dataset with data augmentation for symptom vectors"""
    
    def __init__(self, X, y, augment=True, augmentation_params=None):
        self.X = X
        self.y = y
        self.augment = augment
        
        # Default augmentation parameters
        self.params = augmentation_params or {
            'drop_prob': 0.2,        # Probability of dropping each symptom
            'noise_prob': 0.1,       # Probability of adding noise
            'noise_level': 0.05,     # Magnitude of noise
            'min_symptoms': 2        # Minimum symptoms to keep
        }
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        
        if self.augment and np.random.random() > 0.5:
            x = self.augment_symptoms(x)
        
        return torch.FloatTensor(x), torch.LongTensor([y])[0]
    
    def augment_symptoms(self, symptom_vector):
        """Apply data augmentation to symptom vector"""
        augmented = symptom_vector.copy()
        
        # Get indices of present symptoms
        present_symptoms = np.where(augmented == 1)[0]
        num_symptoms = len(present_symptoms)
        
        # 1. Random symptom dropout (simulate incomplete reporting)
        if num_symptoms > self.params['min_symptoms']:
            drop_mask = np.random.random(num_symptoms) > self.params['drop_prob']
            keep_symptoms = present_symptoms[drop_mask]
            
            # Ensure minimum symptoms
            if len(keep_symptoms) < self.params['min_symptoms']:
                keep_symptoms = np.random.choice(
                    present_symptoms, 
                    self.params['min_symptoms'], 
                    replace=False
                )
            
            # Reset and apply dropout
            augmented = np.zeros_like(augmented)
            augmented[keep_symptoms] = 1
        
        # 2. Add random noise (simulate reporting uncertainty)
        if np.random.random() < self.params['noise_prob']:
            noise = np.random.normal(0, self.params['noise_level'], augmented.shape)
            augmented = augmented + noise
            augmented = np.clip(augmented, 0, 1)
        
        # 3. Random symptom addition (simulate related symptoms)
        if np.random.random() < 0.1:  # 10% chance
            # Add 1-2 random symptoms
            num_to_add = np.random.randint(1, 3)
            absent_symptoms = np.where(augmented == 0)[0]
            
            if len(absent_symptoms) > 0:
                add_indices = np.random.choice(
                    absent_symptoms, 
                    min(num_to_add, len(absent_symptoms)), 
                    replace=False
                )
                augmented[add_indices] = 0.5  # Lower confidence for added symptoms
        
        return augmented


class MixupDataAugmentation:
    """Mixup augmentation for symptom data"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def mixup_data(self, x, y):
        """Apply mixup augmentation"""
        batch_size = x.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Compute mixup loss"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SymptomAugmentationPipeline:
    """Complete augmentation pipeline for training"""
    
    def __init__(self, symptom_vocab):
        self.symptom_vocab = symptom_vocab
        self.symptom_names = list(symptom_vocab.keys())
        
    def get_related_symptoms(self, symptom_idx):
        """Get potentially related symptoms (simple heuristic)"""
        # This is a placeholder - ideally use a symptom knowledge graph
        symptom_name = self.symptom_names[symptom_idx]
        related = []
        
        # Find symptoms with similar prefixes or suffixes
        for idx, name in enumerate(self.symptom_names):
            if idx != symptom_idx:
                if any(word in name for word in symptom_name.split('_')):
                    related.append(idx)
        
        return related
    
    def intelligent_symptom_addition(self, symptom_vector, add_prob=0.15):
        """Add related symptoms intelligently"""
        augmented = symptom_vector.copy()
        present_symptoms = np.where(augmented == 1)[0]
        
        for symptom_idx in present_symptoms:
            if np.random.random() < add_prob:
                related = self.get_related_symptoms(symptom_idx)
                if related:
                    add_idx = np.random.choice(related)
                    augmented[add_idx] = np.random.uniform(0.3, 0.7)
        
        return augmented


# Usage example in training loop
def create_augmented_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create dataloaders with augmentation"""
    
    # Training set with augmentation
    train_dataset = AugmentedSymptomDataset(
        X_train, y_train, 
        augment=True,
        augmentation_params={
            'drop_prob': 0.2,
            'noise_prob': 0.1,
            'noise_level': 0.05,
            'min_symptoms': 2
        }
    )
    
    # Validation set without augmentation
    val_dataset = AugmentedSymptomDataset(
        X_val, y_val, 
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader