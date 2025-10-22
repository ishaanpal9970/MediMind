# ============================================================================
# 4. inference.py - Prediction and Recommendation System
# ============================================================================

import torch
import pickle
import numpy as np
from src.model import AdvancedDiseaseClassifier
from difflib import get_close_matches

class HealthcareAssistant:
    def __init__(self, model_path='models/best_model.pth', 
                 encoder_path='data/processed/encoders.pkl'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load encoders and vocabulary
        with open(encoder_path, 'rb') as f:
            data = pickle.load(f)
            self.disease_encoder = data['disease_encoder']
            self.symptom_vocab = data['symptom_vocab']
            self.disease_info = data['disease_info']
        
        # Load model
        input_size = len(self.symptom_vocab)
        num_diseases = len(self.disease_encoder.classes_)
        
        self.model = AdvancedDiseaseClassifier(input_size, num_diseases)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Symptoms in vocabulary: {len(self.symptom_vocab)}")
        print(f"Diseases: {num_diseases}")

    def normalize_symptom(self, symptom):
       """Normalize and fuzzy match symptom name to dataset vocabulary"""
       clean = symptom.strip().lower().replace(" ", "_")
       if clean in self.symptom_vocab:
           return clean
       # Try fuzzy matching to catch near matches like 'fever' vs 'high_fever'
       close = get_close_matches(clean, self.symptom_vocab.keys(), n=1, cutoff=0.75)
       if close:
           return close[0]
       return None
    
    def predict_disease(self, symptoms_list, top_k=3):
        """Predict disease from list of symptoms"""
        # Create symptom vector
        symptom_vector = np.zeros(len(self.symptom_vocab))
        
        found_symptoms = []
        not_found = []
        
        for symptom in symptoms_list:
           mapped = self.normalize_symptom(symptom)
           if mapped:
               symptom_vector[self.symptom_vocab[mapped]] = 1
               found_symptoms.append(mapped)
           else:
               not_found.append(symptom)
        
        if not_found:
            print(f"Warning: Symptoms not found in database: {not_found}")
        
        # Convert to tensor and predict
        X = torch.FloatTensor(symptom_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(X)
            probabilities = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Get predictions
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            disease = self.disease_encoder.inverse_transform([idx.item()])[0]
            confidence = prob.item() * 100
            
            predictions.append({
                'disease': disease,
                'confidence': confidence,
                'description': self.disease_info.get(disease, {}).get('description', 'N/A'),
                'precautions': self.disease_info.get(disease, {}).get('precautions', [])
            })
        
        return predictions, found_symptoms
    
    def get_available_symptoms(self):
        """Return list of all available symptoms"""
        return sorted(list(self.symptom_vocab.keys()))
