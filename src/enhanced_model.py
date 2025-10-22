import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BERTSymptomEncoder(nn.Module):
    """BERT-based encoder for semantic symptom understanding"""
    def __init__(self, symptom_vocab, embedding_dim=768, freeze_bert=True):
        super(BERTSymptomEncoder, self).__init__()
        
        # Load BioBERT or ClinicalBERT for medical domain
        # Fallback to standard BERT if medical models not available
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            print("Loaded Bio_ClinicalBERT for medical semantic understanding")
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
            print("Loaded standard BERT (consider using Bio_ClinicalBERT for better medical understanding)")
        
        # Freeze BERT weights for faster training (optional)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.symptom_vocab = symptom_vocab
        self.embedding_dim = embedding_dim
        
        # Project BERT embeddings to match model dimensions
        self.projection = nn.Linear(embedding_dim, len(symptom_vocab))
        
    def encode_symptoms(self, symptom_list):
        """Encode list of symptom strings using BERT"""
        # Prepare text for BERT
        symptom_text = " [SEP] ".join([s.replace("_", " ") for s in symptom_list])
        
        # Tokenize and encode
        inputs = self.tokenizer(
            symptom_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(next(self.bert.parameters()).device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def forward(self, symptom_text_list):
        """Forward pass for batch of symptom descriptions"""
        embeddings = []
        for text in symptom_text_list:
            emb = self.encode_symptoms(text)
            embeddings.append(emb)
        
        embeddings = torch.cat(embeddings, dim=0)
        projected = self.projection(embeddings)
        
        return projected


class HybridDiseaseClassifier(nn.Module):
    """Hybrid model combining traditional binary vectors with BERT embeddings"""
    def __init__(self, input_size, num_diseases, use_bert=True, symptom_vocab=None):
        super(HybridDiseaseClassifier, self).__init__()
        
        self.use_bert = use_bert
        
        # Traditional binary input path
        self.binary_encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # BERT semantic path (optional)
        if use_bert and symptom_vocab:
            self.bert_encoder = BERTSymptomEncoder(symptom_vocab, freeze_bert=True)
            self.bert_projection = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            combined_size = 1024  # 512 + 512
        else:
            combined_size = 512
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(128, num_diseases)
        
    def forward(self, x, symptom_texts=None):
        # Binary feature path
        binary_features = self.binary_encoder(x)
        
        # BERT semantic path
        if self.use_bert and symptom_texts is not None:
            bert_output = self.bert_encoder(symptom_texts)
            bert_features = self.bert_projection(bert_output)
            
            # Combine both representations
            combined = torch.cat([binary_features, bert_features], dim=1)
        else:
            combined = binary_features
        
        # Final classification
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output


class TFIDFSymptomEncoder:
    """TF-IDF based symptom encoder for lightweight semantic similarity"""
    def __init__(self, symptom_vocab):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.symptom_vocab = symptom_vocab
        self.symptom_names = list(symptom_vocab.keys())
        
        # Create TF-IDF vectorizer on symptom names
        self.vectorizer = TfidfVectorizer(
            max_features=256,
            ngram_range=(1, 2),
            analyzer='char_wb'  # Character n-grams for partial matching
        )
        
        # Fit on symptom vocabulary
        symptom_texts = [s.replace("_", " ") for s in self.symptom_names]
        self.tfidf_matrix = self.vectorizer.fit_transform(symptom_texts)
        
    def find_similar_symptoms(self, query_symptom, top_k=3):
        """Find similar symptoms using TF-IDF cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_text = query_symptom.replace("_", " ")
        query_vec = self.vectorizer.transform([query_text])
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = [
            (self.symptom_names[i], similarities[i]) 
            for i in top_indices
        ]
        
        return results
    
    def encode_with_similarity(self, symptom_list, similarity_threshold=0.5):
        """Create enhanced binary vector with similar symptoms"""
        import numpy as np
        
        symptom_vector = np.zeros(len(self.symptom_vocab))
        
        for symptom in symptom_list:
            # Direct match
            if symptom in self.symptom_vocab:
                symptom_vector[self.symptom_vocab[symptom]] = 1.0
            else:
                # Find similar symptoms
                similar = self.find_similar_symptoms(symptom, top_k=3)
                for sim_symptom, score in similar:
                    if score >= similarity_threshold:
                        idx = self.symptom_vocab[sim_symptom]
                        symptom_vector[idx] = max(symptom_vector[idx], score)
        
        return symptom_vector
