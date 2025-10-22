# prepare_datasets.py - Prepare all required datasets

import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import json

def create_disease_symptoms_dataset():
    """
    Create comprehensive disease-symptom dataset
    You can download the base dataset from:
    https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
    """
    
    # Extended disease-symptom mapping
    diseases_symptoms = {
        'Common Cold': ['runny nose', 'sneezing', 'sore throat', 'cough', 'mild fever', 'body ache'],
        'Flu (Influenza)': ['high fever', 'severe body ache', 'headache', 'fatigue', 'cough', 'sore throat'],
        'COVID-19': ['fever', 'dry cough', 'loss of taste', 'loss of smell', 'fatigue', 'breathing difficulty'],
        'Malaria': ['high fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting', 'muscle pain'],
        'Dengue': ['high fever', 'severe headache', 'pain behind eyes', 'joint pain', 'muscle pain', 'rash'],
        'Typhoid': ['prolonged fever', 'weakness', 'stomach pain', 'headache', 'loss of appetite'],
        'Pneumonia': ['chest pain', 'cough with phlegm', 'fever', 'chills', 'breathing difficulty'],
        'Tuberculosis': ['persistent cough', 'cough with blood', 'chest pain', 'fever', 'night sweats', 'weight loss'],
        'Asthma': ['breathing difficulty', 'wheezing', 'chest tightness', 'shortness of breath', 'cough'],
        'Migraine': ['severe headache', 'nausea', 'sensitivity to light', 'sensitivity to sound', 'visual disturbances'],
        'Diabetes': ['increased thirst', 'frequent urination', 'extreme hunger', 'fatigue', 'blurred vision'],
        'Hypertension': ['headache', 'dizziness', 'nosebleeds', 'chest pain', 'vision problems'],
        'Heart Attack': ['chest pain', 'pain in arm', 'shortness of breath', 'nausea', 'cold sweat', 'lightheadedness'],
        'Stroke': ['sudden numbness', 'confusion', 'trouble speaking', 'vision problems', 'severe headache', 'loss of balance'],
        'Gastritis': ['stomach pain', 'nausea', 'vomiting', 'loss of appetite', 'bloating', 'burning sensation'],
        'Peptic Ulcer': ['burning stomach pain', 'bloating', 'heartburn', 'nausea', 'dark stools'],
        'Hepatitis': ['jaundice', 'fatigue', 'abdominal pain', 'loss of appetite', 'nausea', 'dark urine'],
        'Appendicitis': ['abdominal pain', 'nausea', 'vomiting', 'loss of appetite', 'fever', 'constipation'],
        'Kidney Stones': ['severe pain in side', 'pain in back', 'painful urination', 'blood in urine', 'nausea'],
        'UTI (Urinary Tract Infection)': ['burning during urination', 'frequent urination', 'cloudy urine', 'pelvic pain'],
        'Chicken Pox': ['itchy rash', 'fever', 'fatigue', 'loss of appetite', 'headache'],
        'Measles': ['fever', 'cough', 'runny nose', 'red eyes', 'rash', 'white spots in mouth'],
        'Arthritis': ['joint pain', 'joint stiffness', 'swelling', 'reduced range of motion'],
        'Osteoporosis': ['back pain', 'stooped posture', 'bone fractures', 'loss of height'],
        'Anemia': ['fatigue', 'weakness', 'pale skin', 'shortness of breath', 'dizziness', 'cold hands'],
        'Thyroid Disorder': ['fatigue', 'weight changes', 'mood swings', 'hair loss', 'irregular heartbeat'],
        'Depression': ['persistent sadness', 'loss of interest', 'fatigue', 'sleep problems', 'difficulty concentrating'],
        'Anxiety': ['excessive worry', 'restlessness', 'rapid heartbeat', 'sweating', 'trembling'],
        'Insomnia': ['difficulty falling asleep', 'waking up frequently', 'daytime fatigue', 'irritability'],
        'Allergy': ['sneezing', 'itchy eyes', 'runny nose', 'rash', 'swelling', 'breathing difficulty'],
        'Bronchitis': ['cough with mucus', 'chest discomfort', 'fatigue', 'shortness of breath', 'slight fever'],
        'Sinusitis': ['facial pain', 'nasal congestion', 'thick nasal discharge', 'reduced sense of smell', 'headache'],
        'Tonsillitis': ['sore throat', 'difficulty swallowing', 'fever', 'swollen tonsils', 'bad breath'],
        'Conjunctivitis': ['red eyes', 'itchy eyes', 'discharge from eyes', 'sensitivity to light', 'blurred vision'],
        'Dermatitis': ['itchy skin', 'red rash', 'dry skin', 'swelling', 'blisters'],
        'Psoriasis': ['red patches of skin', 'silvery scales', 'dry cracked skin', 'itching', 'burning sensation'],
        'Eczema': ['itchy skin', 'red inflamed skin', 'dry scaly skin', 'small bumps', 'thickened skin'],
        'Acne': ['pimples', 'blackheads', 'whiteheads', 'oily skin', 'scarring'],
        'Constipation': ['difficulty passing stools', 'hard stools', 'abdominal pain', 'bloating'],
        'Diarrhea': ['loose stools', 'frequent bowel movements', 'abdominal cramps', 'nausea', 'dehydration'],
        'IBS (Irritable Bowel Syndrome)': ['abdominal pain', 'bloating', 'gas', 'diarrhea', 'constipation'],
        'GERD (Acid Reflux)': ['heartburn', 'chest pain', 'difficulty swallowing', 'regurgitation', 'sour taste'],
        'Food Poisoning': ['nausea', 'vomiting', 'diarrhea', 'abdominal cramps', 'fever'],
        'Vertigo': ['dizziness', 'spinning sensation', 'loss of balance', 'nausea', 'vomiting'],
        'Sciatica': ['lower back pain', 'pain radiating down leg', 'numbness', 'tingling', 'muscle weakness'],
        'Carpal Tunnel Syndrome': ['hand numbness', 'tingling in fingers', 'weakness in hand', 'pain in wrist'],
        'Tennis Elbow': ['elbow pain', 'weak grip', 'pain when lifting', 'tenderness in elbow'],
        'Plantar Fasciitis': ['heel pain', 'foot pain', 'pain in morning', 'difficulty walking'],
        'Gout': ['sudden joint pain', 'swelling', 'redness', 'warmth in joint', 'limited range of motion'],
        'Varicose Veins': ['visible twisted veins', 'leg pain', 'heavy legs', 'swelling', 'itching']
    }
    
    # Generate training data with variations
    data = []
    for disease, symptoms in diseases_symptoms.items():
        # Generate multiple symptom combinations
        for _ in range(20):  # 20 variations per disease
            # Select 3-5 random symptoms
            num_symptoms = np.random.randint(3, min(6, len(symptoms)+1))
            selected_symptoms = np.random.choice(symptoms, size=num_symptoms, replace=False)
            
            # Create natural language description
            symptom_text = ', '.join(selected_symptoms)
            
            data.append({
                'symptoms': symptom_text,
                'disease': disease
            })
    
    df = pd.DataFrame(data)
    df.to_csv('data/disease_symptoms.csv', index=False)
    print(f"Created disease_symptoms.csv with {len(df)} records")
    return df

def create_medicines_dataset():
    """Create medicine recommendations dataset"""
    
    medicines_data = [
        # Common Cold
        {'disease': 'Common Cold', 'medicine_name': 'Paracetamol', 'dosage': '500mg', 
         'frequency': 'Three times daily', 'duration': '3-5 days', 
         'precautions': 'Take after meals. Do not exceed 4g per day'},
        {'disease': 'Common Cold', 'medicine_name': 'Cetirizine', 'dosage': '10mg', 
         'frequency': 'Once daily', 'duration': '3-5 days', 
         'precautions': 'May cause drowsiness. Take at bedtime'},
        {'disease': 'Common Cold', 'medicine_name': 'Vitamin C', 'dosage': '500mg', 
         'frequency': 'Twice daily', 'duration': '7 days', 
         'precautions': 'Take with water after meals'},
        
        # Flu
        {'disease': 'Flu (Influenza)', 'medicine_name': 'Paracetamol', 'dosage': '650mg', 
         'frequency': 'Every 6 hours', 'duration': '5-7 days', 
         'precautions': 'Take with food. Stay hydrated'},
        {'disease': 'Flu (Influenza)', 'medicine_name': 'Ibuprofen', 'dosage': '400mg', 
         'frequency': 'Three times daily', 'duration': '5 days', 
         'precautions': 'Take after meals. Avoid if stomach ulcers'},
        
        # COVID-19
        {'disease': 'COVID-19', 'medicine_name': 'Paracetamol', 'dosage': '500mg', 
         'frequency': 'Every 6 hours if fever', 'duration': 'As needed', 
         'precautions': 'Monitor oxygen levels. Seek medical help if breathing difficulty'},
        {'disease': 'COVID-19', 'medicine_name': 'Vitamin D3', 'dosage': '2000 IU', 
         'frequency': 'Once daily', 'duration': '14 days', 
         'precautions': 'Consult doctor for proper dosage'},
        
        # Malaria
        {'disease': 'Malaria', 'medicine_name': 'Artemether + Lumefantrine', 'dosage': 'As prescribed', 
         'frequency': 'Twice daily', 'duration': '3 days', 
         'precautions': 'Complete full course. Take with fatty food'},
        
        # Dengue
        {'disease': 'Dengue', 'medicine_name': 'Paracetamol', 'dosage': '500mg', 
         'frequency': 'Every 6 hours', 'duration': 'As needed', 
         'precautions': 'Avoid aspirin and ibuprofen. Monitor platelet count'},
        
        # Gastritis
        {'disease': 'Gastritis', 'medicine_name': 'Omeprazole', 'dosage': '20mg', 
         'frequency': 'Once daily before breakfast', 'duration': '14 days', 
         'precautions': 'Take 30 minutes before meal. Avoid spicy food'},
        {'disease': 'Gastritis', 'medicine_name': 'Antacid (Gelusil)', 'dosage': '2 tablets', 
         'frequency': 'After meals', 'duration': '7 days', 
         'precautions': 'Chew tablets thoroughly'},
        
        # Headache/Migraine
        {'disease': 'Migraine', 'medicine_name': 'Sumatriptan', 'dosage': '50mg', 
         'frequency': 'At onset of migraine', 'duration': 'As needed', 
         'precautions': 'Rest in dark room. Avoid triggers'},
        {'disease': 'Migraine', 'medicine_name': 'Paracetamol', 'dosage': '1000mg', 
         'frequency': 'When needed', 'duration': 'As needed', 
         'precautions': 'Do not exceed 4g per day'},
        
        # Diabetes
        {'disease': 'Diabetes', 'medicine_name': 'Metformin', 'dosage': '500mg', 
         'frequency': 'Twice daily with meals', 'duration': 'Continuous', 
         'precautions': 'Monitor blood sugar. Follow diet plan'},
        
        # Hypertension
        {'disease': 'Hypertension', 'medicine_name': 'Amlodipine', 'dosage': '5mg', 
         'frequency': 'Once daily', 'duration': 'Continuous', 
         'precautions': 'Monitor BP regularly. Reduce salt intake'},
        
        # UTI
        {'disease': 'UTI (Urinary Tract Infection)', 'medicine_name': 'Nitrofurantoin', 'dosage': '100mg', 
         'frequency': 'Twice daily', 'duration': '5-7 days', 
         'precautions': 'Complete full course. Drink plenty of water'},
        
        # Allergy
        {'disease': 'Allergy', 'medicine_name': 'Loratadine', 'dosage': '10mg', 
         'frequency': 'Once daily', 'duration': 'As needed', 
         'precautions': 'Non-drowsy formulation'},
        {'disease': 'Allergy', 'medicine_name': 'Cetirizine', 'dosage': '10mg', 
         'frequency': 'Once daily at bedtime', 'duration': 'As needed', 
         'precautions': 'May cause drowsiness'},
        
        # Asthma
        {'disease': 'Asthma', 'medicine_name': 'Salbutamol Inhaler', 'dosage': '2 puffs', 
         'frequency': 'When needed', 'duration': 'As needed', 
         'precautions': 'Carry inhaler always. Rinse mouth after use'},
        
        # Constipation
        {'disease': 'Constipation', 'medicine_name': 'Lactulose', 'dosage': '15ml', 
         'frequency': 'Twice daily', 'duration': '3-5 days', 
         'precautions': 'Increase fiber and water intake'},
        
        # Diarrhea
        {'disease': 'Diarrhea', 'medicine_name': 'Loperamide', 'dosage': '2mg', 
         'frequency': 'After each loose stool', 'duration': 'Max 3 days', 
         'precautions': 'Stay hydrated. Use ORS'},
        {'disease': 'Diarrhea', 'medicine_name': 'Oral Rehydration Solution', 'dosage': '200ml', 
         'frequency': 'After each loose stool', 'duration': 'Until recovered', 
         'precautions': 'Prepare fresh solution'},
        
        # Arthritis
        {'disease': 'Arthritis', 'medicine_name': 'Diclofenac', 'dosage': '50mg', 
         'frequency': 'Twice daily', 'duration': 'As prescribed', 
         'precautions': 'Take after meals. Monitor for side effects'},
        
        # Anemia
        {'disease': 'Anemia', 'medicine_name': 'Ferrous Sulfate', 'dosage': '325mg', 
         'frequency': 'Once daily', 'duration': '3 months', 
         'precautions': 'Take with Vitamin C for better absorption'},
        
        # Thyroid
        {'disease': 'Thyroid Disorder', 'medicine_name': 'Levothyroxine', 'dosage': '50mcg', 
         'frequency': 'Once daily on empty stomach', 'duration': 'Continuous', 
         'precautions': 'Take 30 minutes before breakfast'},
        
        # Acne
        {'disease': 'Acne', 'medicine_name': 'Benzoyl Peroxide Gel', 'dosage': 'Apply thin layer', 
         'frequency': 'Once daily at night', 'duration': '8-12 weeks', 
         'precautions': 'May cause dryness. Use moisturizer'},
        
        # GERD
        {'disease': 'GERD (Acid Reflux)', 'medicine_name': 'Pantoprazole', 'dosage': '40mg', 
         'frequency': 'Once daily before breakfast', 'duration': '4-8 weeks', 
         'precautions': 'Avoid lying down after meals'},
    ]
    
    df = pd.DataFrame(medicines_data)
    df.to_csv('data/medicines.csv', index=False)
    print(f"Created medicines.csv with {len(df)} records")
    return df

def create_precautions_dataset():
    """Create disease precautions dataset"""
    
    precautions_data = [
        {'disease': 'Common Cold', 'precaution_1': 'Rest well', 'precaution_2': 'Drink warm fluids', 
         'precaution_3': 'Use steam inhalation', 'precaution_4': 'Maintain hygiene'},
        {'disease': 'Flu (Influenza)', 'precaution_1': 'Complete bed rest', 'precaution_2': 'Stay hydrated', 
         'precaution_3': 'Isolate from others', 'precaution_4': 'Cover mouth when coughing'},
        {'disease': 'COVID-19', 'precaution_1': 'Self-isolate immediately', 'precaution_2': 'Monitor oxygen levels', 
         'precaution_3': 'Wear mask', 'precaution_4': 'Seek medical help if severe'},
        {'disease': 'Malaria', 'precaution_1': 'Use mosquito nets', 'precaution_2': 'Complete full medication course', 
         'precaution_3': 'Stay in cool environment', 'precaution_4': 'Drink plenty of fluids'},
        {'disease': 'Dengue', 'precaution_1': 'Drink plenty of fluids', 'precaution_2': 'Monitor platelet count', 
         'precaution_3': 'Complete bed rest', 'precaution_4': 'Avoid aspirin and ibuprofen'},
        {'disease': 'Typhoid', 'precaution_1': 'Drink clean water only', 'precaution_2': 'Maintain hygiene', 
         'precaution_3': 'Rest completely', 'precaution_4': 'Eat light easily digestible food'},
        {'disease': 'Pneumonia', 'precaution_1': 'Complete antibiotic course', 'precaution_2': 'Rest adequately', 
         'precaution_3': 'Stay hydrated', 'precaution_4': 'Avoid smoking'},
        {'disease': 'Tuberculosis', 'precaution_1': 'Complete 6-9 month treatment', 'precaution_2': 'Wear mask', 
         'precaution_3': 'Maintain good ventilation', 'precaution_4': 'Nutritious diet'},
        {'disease': 'Asthma', 'precaution_1': 'Avoid triggers', 'precaution_2': 'Carry inhaler always', 
         'precaution_3': 'Regular breathing exercises', 'precaution_4': 'Avoid smoking and pollution'},
        {'disease': 'Migraine', 'precaution_1': 'Identify and avoid triggers', 'precaution_2': 'Maintain sleep schedule', 
         'precaution_3': 'Stay hydrated', 'precaution_4': 'Practice stress management'},
        {'disease': 'Diabetes', 'precaution_1': 'Monitor blood sugar regularly', 'precaution_2': 'Follow diet plan', 
         'precaution_3': 'Exercise daily', 'precaution_4': 'Take medicines on time'},
        {'disease': 'Hypertension', 'precaution_1': 'Reduce salt intake', 'precaution_2': 'Exercise regularly', 
         'precaution_3': 'Manage stress', 'precaution_4': 'Monitor BP daily'},
        {'disease': 'Heart Attack', 'precaution_1': 'Call emergency immediately', 'precaution_2': 'Chew aspirin if available', 
         'precaution_3': 'Stay calm and rest', 'precaution_4': 'Do not drive yourself'},
        {'disease': 'Stroke', 'precaution_1': 'Call emergency immediately', 'precaution_2': 'Note time of symptom onset', 
         'precaution_3': 'Do not give food or water', 'precaution_4': 'Keep patient calm'},
        {'disease': 'Gastritis', 'precaution_1': 'Avoid spicy food', 'precaution_2': 'Eat small frequent meals', 
         'precaution_3': 'Avoid alcohol', 'precaution_4': 'Reduce stress'},
        {'disease': 'Peptic Ulcer', 'precaution_1': 'Avoid NSAIDs', 'precaution_2': 'Quit smoking', 
         'precaution_3': 'Avoid spicy food', 'precaution_4': 'Complete medication course'},
        {'disease': 'Hepatitis', 'precaution_1': 'Complete bed rest', 'precaution_2': 'Avoid alcohol completely', 
         'precaution_3': 'Eat nutritious food', 'precaution_4': 'Maintain hygiene'},
        {'disease': 'Appendicitis', 'precaution_1': 'Seek immediate medical attention', 'precaution_2': 'Do not eat or drink', 
         'precaution_3': 'Avoid pain medications', 'precaution_4': 'Do not apply heat'},
        {'disease': 'Kidney Stones', 'precaution_1': 'Drink plenty of water', 'precaution_2': 'Reduce salt intake', 
         'precaution_3': 'Limit animal protein', 'precaution_4': 'Avoid stone-forming foods'},
        {'disease': 'UTI (Urinary Tract Infection)', 'precaution_1': 'Drink plenty of water', 'precaution_2': 'Urinate frequently', 
         'precaution_3': 'Maintain hygiene', 'precaution_4': 'Complete antibiotic course'},
        {'disease': 'Chicken Pox', 'precaution_1': 'Isolate from others', 'precaution_2': 'Do not scratch', 
         'precaution_3': 'Trim nails', 'precaution_4': 'Use calamine lotion'},
        {'disease': 'Measles', 'precaution_1': 'Complete isolation', 'precaution_2': 'Rest in dark room', 
         'precaution_3': 'Stay hydrated', 'precaution_4': 'Maintain hygiene'},
        {'disease': 'Arthritis', 'precaution_1': 'Regular gentle exercise', 'precaution_2': 'Maintain healthy weight', 
         'precaution_3': 'Apply heat/cold therapy', 'precaution_4': 'Protect joints'},
        {'disease': 'Anemia', 'precaution_1': 'Eat iron-rich foods', 'precaution_2': 'Take prescribed supplements', 
         'precaution_3': 'Increase Vitamin C intake', 'precaution_4': 'Regular blood tests'},
        {'disease': 'Allergy', 'precaution_1': 'Identify and avoid allergens', 'precaution_2': 'Keep antihistamine handy', 
         'precaution_3': 'Maintain clean environment', 'precaution_4': 'Consider allergy testing'},
        {'disease': 'Bronchitis', 'precaution_1': 'Avoid smoking', 'precaution_2': 'Use humidifier', 
         'precaution_3': 'Stay hydrated', 'precaution_4': 'Get adequate rest'},
        {'disease': 'Constipation', 'precaution_1': 'Increase fiber intake', 'precaution_2': 'Drink more water', 
         'precaution_3': 'Exercise regularly', 'precaution_4': 'Establish routine'},
        {'disease': 'Diarrhea', 'precaution_1': 'Stay hydrated with ORS', 'precaution_2': 'Eat bland foods', 
         'precaution_3': 'Avoid dairy products', 'precaution_4': 'Maintain hygiene'},
        {'disease': 'GERD (Acid Reflux)', 'precaution_1': 'Avoid trigger foods', 'precaution_2': 'Eat small meals', 
         'precaution_3': 'Do not lie down after eating', 'precaution_4': 'Elevate head while sleeping'},
    ]
    
    df = pd.DataFrame(precautions_data)
    df.to_csv('data/precautions.csv', index=False)
    print(f"Created precautions.csv with {len(df)} records")
    return df

def create_sample_hospitals():
    """Create sample hospital data for different cities"""
    
    hospitals_data = [
        # Delhi
        {'name': 'AIIMS Delhi', 'city': 'Delhi', 'address': 'Ansari Nagar, New Delhi', 
         'phone': '+91-11-26588500', 'latitude': 28.5672, 'longitude': 77.2100, 
         'emergency_available': True, 'rating': 4.5},
        {'name': 'Safdarjung Hospital', 'city': 'Delhi', 'address': 'Safdarjung, New Delhi', 
         'phone': '+91-11-26165060', 'latitude': 28.5678, 'longitude': 77.2065, 
         'emergency_available': True, 'rating': 4.2},
        
        # Mumbai
        {'name': 'KEM Hospital', 'city': 'Mumbai', 'address': 'Parel, Mumbai', 
         'phone': '+91-22-24107000', 'latitude': 19.0030, 'longitude': 72.8430, 
         'emergency_available': True, 'rating': 4.3},
        {'name': 'Lilavati Hospital', 'city': 'Mumbai', 'address': 'Bandra West, Mumbai', 
         'phone': '+91-22-26567891', 'latitude': 19.0596, 'longitude': 72.8295, 
         'emergency_available': True, 'rating': 4.6},
        
        # Bangalore
        {'name': 'Manipal Hospital', 'city': 'Bangalore', 'address': 'HAL Airport Road, Bangalore', 
         'phone': '+91-80-25024444', 'latitude': 12.9576, 'longitude': 77.6365, 
         'emergency_available': True, 'rating': 4.4},
        {'name': 'Victoria Hospital', 'city': 'Bangalore', 'address': 'Fort, Bangalore', 
         'phone': '+91-80-26700301', 'latitude': 12.9716, 'longitude': 77.5946, 
         'emergency_available': True, 'rating': 4.1},
        
        # Chennai
        {'name': 'Apollo Hospital', 'city': 'Chennai', 'address': 'Greams Road, Chennai', 
         'phone': '+91-44-28293333', 'latitude': 13.0569, 'longitude': 80.2433, 
         'emergency_available': True, 'rating': 4.5},
        
        # Kolkata
        {'name': 'SSKM Hospital', 'city': 'Kolkata', 'address': '244 AJC Bose Road, Kolkata', 
         'phone': '+91-33-22231200', 'latitude': 22.5469, 'longitude': 88.3515, 
         'emergency_available': True, 'rating': 4.0},
    ]
    
    df = pd.DataFrame(hospitals_data)
    df.to_csv('data/hospitals.csv', index=False)
    print(f"Created hospitals.csv with {len(df)} records")
    return df

def main():
    """Run all data preparation"""
    print("="*60)
    print("MediMind Data Preparation")
    print("="*60)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Create all datasets
    print("\n1. Creating Disease-Symptoms Dataset...")
    create_disease_symptoms_dataset()
    
    print("\n2. Creating Medicines Dataset...")
    create_medicines_dataset()
    
    print("\n3. Creating Precautions Dataset...")
    create_precautions_dataset()
    
    print("\n4. Creating Hospitals Dataset...")
    create_sample_hospitals()
    
    print("\n" + "="*60)
    print("All datasets created successfully!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Run: python train_model.py")
    print("2. Run: python app.py")
    print("3. Open frontend in browser")

if __name__ == "__main__":
    main()