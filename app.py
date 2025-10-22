import streamlit as st
import torch
import pickle
import numpy as np
from src.inference import HealthcareAssistant
from src.enhanced_model import TFIDFSymptomEncoder
import time

# Page configuration
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1f77b4;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4caf50;
    }
    .disease-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []
if 'assistant' not in st.session_state:
    with st.spinner('Loading AI models...'):
        st.session_state.assistant = HealthcareAssistant()
        # Load TF-IDF encoder for semantic similarity
        with open('data/processed/encoders.pkl', 'rb') as f:
            data = pickle.load(f)
            st.session_state.tfidf_encoder = TFIDFSymptomEncoder(data['symptom_vocab'])

def add_message(role, content):
    """Add message to chat history"""
    st.session_state.chat_history.append({
        'role': role,
        'content': content,
        'timestamp': time.strftime('%H:%M:%S')
    })

def display_chat_history():
    """Display chat messages"""
    for msg in st.session_state.chat_history:
        css_class = "user-message" if msg['role'] == 'user' else "bot-message"
        icon = "👤" if msg['role'] == 'user' else "🤖"
        
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{icon} {msg['role'].title()}</strong> 
            <small style="color: gray;">{msg['timestamp']}</small>
            <div style="margin-top: 0.5rem;">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

def display_predictions(predictions, symptoms_found):
    """Display disease predictions with detailed information"""
    
    st.markdown("### 🔍 Analysis Results")
    
    # Display recognized symptoms
    st.markdown("**Recognized Symptoms:**")
    symptom_tags = " ".join([f"`{s}`" for s in symptoms_found])
    st.markdown(symptom_tags)
    
    st.markdown("---")
    
    # Display top predictions
    for i, pred in enumerate(predictions, 1):
        confidence = pred['confidence']
        disease = pred['disease']
        description = pred['description']
        precautions = pred['precautions']
        
        # Color code by confidence
        if confidence >= 70:
            color = "#4caf50"
            confidence_text = "High Confidence"
        elif confidence >= 40:
            color = "#ff9800"
            confidence_text = "Moderate Confidence"
        else:
            color = "#f44336"
            confidence_text = "Low Confidence"
        
        st.markdown(f"""
        <div class="disease-card">
            <h3 style="color: {color};">#{i} {disease}</h3>
            <p><strong>{confidence_text}:</strong> {confidence:.1f}%</p>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; width: 100%;">
                <div style="background-color: {color}; width: {confidence}%; height: 100%; border-radius: 10px;"></div>
            </div>
            <p style="margin-top: 1rem;"><strong>Description:</strong> {description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if precautions:
            st.markdown("**Recommended Precautions:**")
            for prec in precautions:
                st.markdown(f"- {prec}")
        
        st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Healthcare Assistant</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This is an AI-based preliminary screening tool. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult a qualified healthcare provider for medical concerns.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Interaction mode
        mode = st.radio(
            "Select Input Mode:",
            ["💬 Chat Mode", "📋 Symptom Selection", "🔍 Smart Search"]
        )
        
        st.markdown("---")
        
        # Top K predictions
        top_k = st.slider("Number of predictions:", 1, 5, 3)
        
        # Confidence threshold
        confidence_threshold = st.slider("Confidence threshold (%):", 0, 100, 30)
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.selected_symptoms = []
            st.rerun()
        
        if st.button("📊 View All Symptoms"):
            with st.expander("Available Symptoms", expanded=True):
                symptoms = st.session_state.assistant.get_available_symptoms()
                st.write(f"Total symptoms: {len(symptoms)}")
                for symptom in symptoms[:50]:  # Show first 50
                    st.text(f"• {symptom}")
                if len(symptoms) > 50:
                    st.text(f"... and {len(symptoms) - 50} more")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Conversation")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            display_chat_history()
        
        # Input area based on mode
        if mode == "💬 Chat Mode":
            st.markdown("---")
            user_input = st.text_input(
                "Describe your symptoms:",
                placeholder="E.g., I have fever, headache, and cough...",
                key="chat_input"
            )
            
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                submit = st.button("🚀 Analyze", type="primary")
            
            if submit and user_input:
                # Add user message
                add_message('user', user_input)
                
                # Parse symptoms from text
                words = user_input.lower().replace(',', ' ').split()
                symptoms = [w.strip() for w in words if len(w) > 3]
                
                with st.spinner('Analyzing symptoms...'):
                    predictions, found = st.session_state.assistant.predict_disease(
                        symptoms, top_k=top_k
                    )
                    
                    # Filter by confidence
                    predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
                    
                    if predictions:
                        bot_response = f"I've analyzed your symptoms. Found {len(found)} recognized symptoms."
                        add_message('bot', bot_response)
                    else:
                        bot_response = "I couldn't identify any diseases with sufficient confidence. Please try describing your symptoms differently."
                        add_message('bot', bot_response)
                
                st.rerun()
        
        elif mode == "📋 Symptom Selection":
            st.markdown("---")
            available_symptoms = st.session_state.assistant.get_available_symptoms()
            
            selected = st.multiselect(
                "Select your symptoms:",
                options=available_symptoms,
                default=st.session_state.selected_symptoms,
                placeholder="Start typing to search symptoms..."
            )
            
            st.session_state.selected_symptoms = selected
            
            if st.button("🚀 Analyze Symptoms", type="primary") and selected:
                add_message('user', f"Selected symptoms: {', '.join(selected)}")
                
                with st.spinner('Analyzing symptoms...'):
                    predictions, found = st.session_state.assistant.predict_disease(
                        selected, top_k=top_k
                    )
                    
                    predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
                    
                    if predictions:
                        bot_response = f"Analysis complete for {len(found)} symptoms."
                        add_message('bot', bot_response)
                    else:
                        bot_response = "No confident predictions. Try selecting more symptoms."
                        add_message('bot', bot_response)
                
                st.rerun()
        
        else:  # Smart Search Mode
            st.markdown("---")
            search_query = st.text_input(
                "Search for similar symptoms:",
                placeholder="Type a symptom (exact match not required)..."
            )
            
            if search_query:
                # Use TF-IDF for fuzzy matching
                similar = st.session_state.tfidf_encoder.find_similar_symptoms(
                    search_query, top_k=5
                )
                
                st.markdown("**Similar symptoms found:**")
                for symptom, score in similar:
                    st.markdown(f"- {symptom} (similarity: {score:.2%})")
                
                # Add selected symptoms
                if st.button("Add All Similar Symptoms"):
                    st.session_state.selected_symptoms.extend([s for s, _ in similar])
                    st.session_state.selected_symptoms = list(set(st.session_state.selected_symptoms))
    
    with col2:
        st.header("📊 Results")
        
        # Display latest predictions if available
        if st.session_state.chat_history:
            last_user_msg = None
            for msg in reversed(st.session_state.chat_history):
                if msg['role'] == 'user':
                    last_user_msg = msg['content']
                    break
            
            if last_user_msg:
                # Extract symptoms
                if mode == "📋 Symptom Selection" or "Selected symptoms:" in last_user_msg:
                    symptoms = st.session_state.selected_symptoms
                else:
                    words = last_user_msg.lower().replace(',', ' ').split()
                    symptoms = [w.strip() for w in words if len(w) > 3]
                
                if symptoms:
                    predictions, found = st.session_state.assistant.predict_disease(
                        symptoms, top_k=top_k
                    )
                    
                    predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
                    
                    if predictions:
                        display_predictions(predictions, found)
                    else:
                        st.info("No predictions available with current settings. Try lowering the confidence threshold.")
        else:
            st.info("Start a conversation to see results here!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <small>Powered by Deep Learning & Natural Language Processing | 
        Made with ❤️ using Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
