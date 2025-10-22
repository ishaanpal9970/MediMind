# app.py - Main Flask Application
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from mongoengine import connect, Document, StringField, FloatField, BooleanField, DateTimeField, IntField, ListField, ReferenceField
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import os
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps
import logging

app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app.config['SECRET_KEY'] = 'c047791a92854a90b6e4d6f80f277edc804fbc5a79f0a13fec3dde339f8352b6'
app.config['MONGODB_SETTINGS'] = {
    'db': 'medimind',
    'host': 'mongodb+srv://ishaan:suzuki5117@cluster0.oefcfam.mongodb.net/medimind'
}

# Connect to MongoDB
connect(**app.config['MONGODB_SETTINGS'])

# ==================== DATABASE MODELS ====================

class User(Document):
    name = StringField(required=True, max_length=100)
    email = StringField(required=True, unique=True, max_length=120)
    password = StringField(required=True, max_length=200)
    age = IntField()
    gender = StringField(max_length=10)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'users'}

class Provider(Document):
    name = StringField(required=True, max_length=200)
    type = StringField(required=True, max_length=50)  # clinic/chemist
    license_number = StringField(required=True, unique=True, max_length=100)
    email = StringField(required=True, unique=True, max_length=120)
    password = StringField(required=True, max_length=200)
    phone = StringField(required=True, max_length=20)
    address = StringField(required=True)
    latitude = FloatField(required=True)
    longitude = FloatField(required=True)
    is_verified = BooleanField(default=False)
    available_24x7 = BooleanField(default=False)
    opening_time = StringField(max_length=10)
    closing_time = StringField(max_length=10)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'providers'}

class Consultation(Document):
    user_id = StringField(required=True)  # Reference to User document ID
    symptoms = StringField(required=True)
    predicted_disease = StringField(max_length=200)
    confidence = FloatField()
    severity = StringField(max_length=20)
    recommended_medicines = StringField()
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'consultations'}

class Hospital(Document):
    name = StringField(required=True, max_length=200)
    address = StringField(required=True)
    phone = StringField(max_length=20)
    latitude = FloatField(required=True)
    longitude = FloatField(required=True)
    emergency_available = BooleanField(default=True)
    rating = FloatField(default=0.0)

    meta = {'collection': 'hospitals'}

# ==================== AUTHENTICATION ====================

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            token = token.split()[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.objects(id=data['user_id']).first()
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# ==================== WELCOME ENDPOINT ====================

@app.route('/api/welcome', methods=['GET'])
def welcome():
    logger.info(f"Request received: {request.method} {request.path}")
    return jsonify({'message': 'Welcome to MediMind API!'}), 200

# ==================== USER ROUTES ====================

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json

    if User.objects(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 400

    hashed_password = generate_password_hash(data['password'])
    new_user = User(
        name=data['name'],
        email=data['email'],
        password=hashed_password,
        age=data.get('age'),
        gender=data.get('gender')
    )

    new_user.save()

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.objects(email=data['email']).first()

    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'message': 'Invalid credentials'}), 401

    token = jwt.encode({
        'user_id': str(user.id),
        'exp': datetime.utcnow().timestamp() + 86400
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        'token': token,
        'user': {
            'id': str(user.id),
            'name': user.name,
            'email': user.email
        }
    }), 200

# ==================== PROVIDER REGISTRATION ====================

@app.route('/api/provider/register', methods=['POST'])
def register_provider():
    data = request.json

    if Provider.objects(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 400

    if Provider.objects(license_number=data['license_number']).first():
        return jsonify({'message': 'License number already registered'}), 400

    hashed_password = generate_password_hash(data['password'])
    new_provider = Provider(
        name=data['name'],
        type=data['type'],
        license_number=data['license_number'],
        email=data['email'],
        password=hashed_password,
        phone=data['phone'],
        address=data['address'],
        latitude=data['latitude'],
        longitude=data['longitude'],
        available_24x7=data.get('available_24x7', False),
        opening_time=data.get('opening_time'),
        closing_time=data.get('closing_time')
    )

    new_provider.save()

    return jsonify({'message': 'Provider registered successfully. Awaiting verification.'}), 201

@app.route('/api/providers/nearby', methods=['POST'])
def get_nearby_providers():
    data = request.json
    user_lat = data['latitude']
    user_lon = data['longitude']
    provider_type = data.get('type', 'clinic')  # clinic or chemist
    radius = data.get('radius', 10)  # km

    providers = Provider.objects(type=provider_type, is_verified=True)

    nearby = []
    for provider in providers:
        distance = calculate_distance(user_lat, user_lon, provider.latitude, provider.longitude)
        if distance <= radius:
            nearby.append({
                'id': str(provider.id),
                'name': provider.name,
                'address': provider.address,
                'phone': provider.phone,
                'distance': round(distance, 2),
                'latitude': provider.latitude,
                'longitude': provider.longitude,
                'available_24x7': provider.available_24x7,
                'opening_time': provider.opening_time,
                'closing_time': provider.closing_time
            })

    nearby.sort(key=lambda x: x['distance'])
    return jsonify(nearby), 200

# ==================== VOICE PROCESSING ====================

@app.route('/api/voice/transcribe', methods=['POST'])
@token_required
def transcribe_voice(current_user):
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
        import io

        # Get audio file from request
        if 'audio' not in request.files:
            return jsonify({'message': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        # Convert audio to WAV format if needed
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))

        # Export as WAV
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format='wav')
        wav_buffer.seek(0)

        # Initialize recognizer
        recognizer = sr.Recognizer()

        # Convert to AudioData
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)

        # Transcribe
        text = recognizer.recognize_google(audio_data)

        return jsonify({'transcription': text}), 200

    except sr.UnknownValueError:
        return jsonify({'message': 'Could not understand audio'}), 400
    except sr.RequestError as e:
        return jsonify({'message': f'Could not request results; {e}'}), 500
    except Exception as e:
        return jsonify({'message': f'Error processing audio: {str(e)}'}), 500

@app.route('/api/voice/speak', methods=['POST'])
@token_required
def speak_text(current_user):
    try:
        import pyttsx3

        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'message': 'No text provided'}), 400

        # Initialize TTS engine
        engine = pyttsx3.init()

        # Set properties
        engine.setProperty('rate', 180)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

        # Save to temporary file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name

        engine.save_to_file(text, temp_filename)
        engine.runAndWait()

        # Read the file and return as response
        with open(temp_filename, 'rb') as f:
            audio_data = f.read()

        # Clean up
        os.unlink(temp_filename)

        from flask import send_file
        import io

        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='response.wav'
        ), 200

    except Exception as e:
        return jsonify({'message': f'Error generating speech: {str(e)}'}), 500

# ==================== SYMPTOM ANALYSIS ====================

@app.route('/api/analyze', methods=['POST'])
@token_required
def analyze_symptoms(current_user):
    data = request.json
    symptoms = data['symptoms']
    
    # Load models
    try:
        with open('models/ensemble.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        return jsonify({'message': 'Models not found. Please train models first.'}), 500
    
    # Preprocess symptoms
    processed_symptoms = preprocess_text(symptoms)
    
    # Vectorize
    symptoms_vector = vectorizer.transform([processed_symptoms])
    
    # Predict
    prediction = model.predict(symptoms_vector)[0]
    prediction_proba = model.predict_proba(symptoms_vector)[0]
    
    disease = label_encoder.inverse_transform([prediction])[0]
    confidence = float(max(prediction_proba) * 100)
    
    # Get severity
    severity = determine_severity(disease, confidence)
    
    # Get medicine recommendations
    medicines = get_medicine_recommendations(disease)
    
    # Save consultation
    consultation = Consultation(
        user_id=str(current_user.id),
        symptoms=symptoms,
        predicted_disease=disease,
        confidence=confidence,
        severity=severity,
        recommended_medicines=str(medicines)
    )
    consultation.save()
    
    response = {
        'disease': disease,
        'confidence': round(confidence, 2),
        'severity': severity,
        'medicines': medicines,
        'precautions': get_precautions(disease)
    }
    
    return jsonify(response), 200

# ==================== HOSPITAL FINDER ====================

@app.route('/api/hospitals/nearby', methods=['POST'])
def get_nearby_hospitals():
    data = request.json
    user_lat = data['lat']
    user_lon = data['lon']
    radius = data.get('radius', 20)  # km

    hospitals = Hospital.objects(emergency_available=True)

    nearby = []
    for hospital in hospitals:
        distance = calculate_distance(user_lat, user_lon, hospital.latitude, hospital.longitude)
        if distance <= radius:
            nearby.append({
                'id': str(hospital.id),
                'name': hospital.name,
                'address': hospital.address,
                'phone': hospital.phone,
                'distance': round(distance, 2),
                'latitude': hospital.latitude,
                'longitude': hospital.longitude,
                'rating': hospital.rating,
                'directions_url': f"https://www.google.com/maps/dir/?api=1&destination={hospital.latitude},{hospital.longitude}"
            })

    nearby.sort(key=lambda x: x['distance'])
    return jsonify(nearby), 200

# ==================== HELPER FUNCTIONS ====================

def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and lemmatize
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    
    return ' '.join(words)

def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def determine_severity(disease, confidence):
    critical_diseases = [
        'Heart Attack', 'Stroke', 'Pneumonia', 'Tuberculosis', 
        'Hepatitis', 'Malaria', 'Dengue', 'COVID-19'
    ]
    
    if disease in critical_diseases and confidence > 70:
        return 'CRITICAL'
    elif confidence > 80:
        return 'HIGH'
    elif confidence > 60:
        return 'MODERATE'
    else:
        return 'LOW'

def get_medicine_recommendations(disease):
    # Load medicine database
    try:
        medicines_df = pd.read_csv('data/medicines.csv')
        disease_medicines = medicines_df[medicines_df['disease'] == disease]
        
        if len(disease_medicines) == 0:
            return []
        
        recommendations = []
        for _, med in disease_medicines.iterrows():
            recommendations.append({
                'name': med['medicine_name'],
                'dosage': med['dosage'],
                'frequency': med['frequency'],
                'duration': med['duration'],
                'precautions': med['precautions']
            })
        
        return recommendations[:3]  # Top 3 medicines
    except:
        return []

def get_precautions(disease):
    try:
        precautions_df = pd.read_csv('data/precautions.csv')
        disease_precautions = precautions_df[precautions_df['disease'] == disease]
        
        if len(disease_precautions) == 0:
            return []
        
        return disease_precautions.iloc[0][['precaution_1', 'precaution_2', 'precaution_3', 'precaution_4']].tolist()
    except:
        return []

# ==================== INITIALIZE DATABASE ====================

# MongoDB handles schema automatically, no need for table creation

# ==================== RUN APPLICATION ====================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
