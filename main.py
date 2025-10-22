from src.data_loader import MedicalDataLoader
from src.train import DiseaseModelTrainer
from src.inference import HealthcareAssistant
from src.enhanced_model import HybridDiseaseClassifier, TFIDFSymptomEncoder
from src.data_augmentation import create_augmented_dataloaders
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def train_with_augmentation(use_bert=False, use_tfidf=True):
    """Training pipeline with data augmentation and semantic models"""
    
    print("="*70)
    print("ADVANCED MEDICAL DIAGNOSIS SYSTEM - TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and Preprocessing Data...")
    data_loader = MedicalDataLoader()
    disease_df, description_df, precaution_df, severity_df = data_loader.load_datasets()
    
    X, y, processed_df = data_loader.preprocess_disease_data(disease_df)
    data_loader.load_disease_info(description_df, precaution_df)
    data_loader.save_preprocessed_data(X, y)
    
    print(f"✓ Dataset shape: X={X.shape}, y={y.shape}")
    print(f"✓ Number of unique diseases: {len(np.unique(y))}")
    print(f"✓ Number of unique symptoms: {X.shape[1]}")
    
    # Step 2: Initialize TF-IDF encoder for semantic similarity
    if use_tfidf:
        print("\n[2/6] Initializing TF-IDF Semantic Encoder...")
        tfidf_encoder = TFIDFSymptomEncoder(data_loader.symptom_vocab)
        print("✓ TF-IDF encoder trained on symptom vocabulary")
        print(f"✓ Can find similar symptoms using cosine similarity")
    
    # Step 3: Split data with stratification
    print("\n[3/6] Splitting Dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train: {X_train.shape[0]} samples")
    print(f"✓ Validation: {X_val.shape[0]} samples")
    print(f"✓ Test: {X_test.shape[0]} samples")
    
    # Step 4: Create DataLoaders with augmentation
    print("\n[4/6] Creating Augmented DataLoaders...")
    train_loader, val_loader = create_augmented_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Test loader without augmentation
    from src.data_augmentation import AugmentedSymptomDataset
    test_dataset = AugmentedSymptomDataset(X_test, y_test, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    print("✓ Training data augmentation enabled:")
    print("  - Random symptom dropout (20%)")
    print("  - Gaussian noise injection (10%)")
    print("  - Related symptom addition (10%)")
    
    # Step 5: Initialize model
    print("\n[5/6] Initializing Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    input_size = X.shape[1]
    num_diseases = len(np.unique(y))
    
    if use_bert:
        print("✓ Initializing Hybrid Model with BERT embeddings...")
        model = HybridDiseaseClassifier(
            input_size, 
            num_diseases, 
            use_bert=True,
            symptom_vocab=data_loader.symptom_vocab
        )
        print("✓ Model will use both binary vectors and BERT semantic embeddings")
    else:
        from src.model import AdvancedDiseaseClassifier
        print("✓ Initializing Advanced Disease Classifier...")
        model = AdvancedDiseaseClassifier(input_size, num_diseases)
    
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 6: Train model
    print("\n[6/6] Training Model with Advanced Techniques...")
    print("=" * 70)
    
    trainer = DiseaseModelTrainer(model, device=device)
    best_val_loss = trainer.train(
        train_loader, 
        val_loader, 
        epochs=100,
        lr=0.001,
        early_stopping_patience=10
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    print(f"✓ Model saved to: models/best_model.pth")
    
    # Step 7: Evaluate on test set
    print("\n[Evaluation] Testing on held-out test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f"✓ Test Accuracy: {test_accuracy:.2f}%")
    
    # Step 8: Plot training history
    trainer.plot_training_history()
    
    # Step 9: Test inference with semantic similarity
    print("\n" + "=" * 70)
    print("TESTING HEALTHCARE ASSISTANT WITH SEMANTIC UNDERSTANDING")
    print("=" * 70)
    
    assistant = HealthcareAssistant()
    
    # Test 1: Exact symptom match
    print("\n[Test 1] Exact symptom matching:")
    test_symptoms = ['fever', 'cough', 'fatigue']
    print(f"Input symptoms: {test_symptoms}")
    
    predictions, found = assistant.predict_disease(test_symptoms, top_k=3)
    print(f"\nRecognized symptoms: {found}")
    print("\nTop 3 predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"\n{i}. {pred['disease']}")
        print(f"   Confidence: {pred['confidence']:.2f}%")
        print(f"   Description: {pred['description'][:100]}...")
    
    # Test 2: Fuzzy matching with TF-IDF
    if use_tfidf:
        print("\n" + "-" * 70)
        print("[Test 2] Semantic similarity search:")
        query_symptom = "high temperature"
        print(f"Searching for symptoms similar to: '{query_symptom}'")
        
        similar_symptoms = tfidf_encoder.find_similar_symptoms(query_symptom, top_k=5)
        print("\nSimilar symptoms found:")
        for symptom, score in similar_symptoms:
            print(f"  - {symptom} (similarity: {score:.3f})")
    
    # Test 3: Partial symptom input
    print("\n" + "-" * 70)
    print("[Test 3] Partial symptom description:")
    partial_symptoms = ['headache', 'nausea']
    print(f"Input symptoms: {partial_symptoms}")
    
    predictions, found = assistant.predict_disease(partial_symptoms, top_k=3)
    print(f"\nRecognized: {found}")
    print("Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['disease']} - {pred['confidence']:.1f}%")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\n✓ Model trained and ready for deployment")
    print("✓ Run the Streamlit app: streamlit run app.py")
    print("=" * 70)


def main():
    """Main execution with command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Advanced Medical Diagnosis System - Training Pipeline'
    )
    parser.add_argument(
        '--use-bert', 
        action='store_true',
        help='Use BERT embeddings for semantic understanding (requires transformers)'
    )
    parser.add_argument(
        '--no-tfidf',
        action='store_true',
        help='Disable TF-IDF semantic similarity'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test without full training'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running quick inference test...")
        assistant = HealthcareAssistant()
        test_symptoms = ['fever', 'cough', 'headache']
        predictions, found = assistant.predict_disease(test_symptoms, top_k=3)
        
        print(f"\nTest symptoms: {test_symptoms}")
        print(f"Found: {found}")
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['disease']} ({pred['confidence']:.1f}%)")
    else:
        train_with_augmentation(
            use_bert=args.use_bert,
            use_tfidf=not args.no_tfidf
        )


if __name__ == '__main__':
    main()