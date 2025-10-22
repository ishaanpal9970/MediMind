# voice_processing.py - Voice Input Processing for MediMind

import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import io
import wave

class VoiceProcessor:
    """Process voice input for symptom analysis"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.setup_voice_engine()
        
    def setup_voice_engine(self):
        """Configure text-to-speech engine"""
        # Set properties
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0-1)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        # Set to female voice if available
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
    
    def speak(self, text):
        """Convert text to speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def listen_from_microphone(self, duration=5):
        """
        Listen to microphone input
        Args:
            duration: Maximum recording duration in seconds
        Returns:
            Transcribed text
        """
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print(f"Listening for {duration} seconds...")
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
                
                print("Processing audio...")
                text = self.transcribe_audio(audio)
                
                return text
                
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
        except sr.UnknownValueError:
            return "Could not understand audio. Please speak clearly."
        except sr.RequestError as e:
            return f"Could not process audio: {e}"
        except Exception as e:
            return f"Error: {e}"
    
    def transcribe_audio(self, audio_data):
        """
        Transcribe audio to text using multiple recognition engines
        Args:
            audio_data: AudioData object from speech_recognition
        Returns:
            Transcribed text
        """
        try:
            # Try Google Speech Recognition (free)
            try:
                text = self.recognizer.recognize_google(audio_data)
                print(f"Google Recognition: {text}")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError:
                print("Google Speech Recognition service unavailable")
            
            # Fallback to Sphinx (offline)
            try:
                text = self.recognizer.recognize_sphinx(audio_data)
                print(f"Sphinx Recognition: {text}")
                return text
            except sr.UnknownValueError:
                print("Sphinx could not understand audio")
            except sr.RequestError as e:
                print(f"Sphinx error: {e}")
            
            return "Could not transcribe audio. Please try again."
            
        except Exception as e:
            return f"Transcription error: {e}"
    
    def process_audio_file(self, audio_file_path):
        """
        Process audio file and convert to text
        Args:
            audio_file_path: Path to audio file (WAV, MP3, etc.)
        Returns:
            Transcribed text
        """
        try:
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
                text = self.transcribe_audio(audio)
                return text
                
        except Exception as e:
            return f"Error processing audio file: {e}"
    
    def process_audio_bytes(self, audio_bytes, sample_rate=16000):
        """
        Process audio from bytes (for web uploads)
        Args:
            audio_bytes: Audio data in bytes
            sample_rate: Sample rate of audio
        Returns:
            Transcribed text
        """
        try:
            # Convert bytes to AudioData
            audio_data = sr.AudioData(audio_bytes, sample_rate, 2)
            text = self.transcribe_audio(audio_data)
            return text
            
        except Exception as e:
            return f"Error processing audio bytes: {e}"
    
    def enhance_audio(self, audio_path, output_path=None):
        """
        Enhance audio quality for better recognition
        Args:
            audio_path: Path to input audio
            output_path: Path to save enhanced audio
        Returns:
            Path to enhanced audio
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Normalize volume
            audio = audio.normalize()
            
            # Remove silence
            chunks = split_on_silence(
                audio,
                min_silence_len=500,  # ms
                silence_thresh=audio.dBFS - 14,
                keep_silence=250
            )
            
            # Combine chunks
            enhanced = AudioSegment.empty()
            for chunk in chunks:
                enhanced += chunk
            
            # Save enhanced audio
            if output_path is None:
                output_path = audio_path.replace('.', '_enhanced.')
            
            enhanced.export(output_path, format="wav")
            return output_path
            
        except Exception as e:
            print(f"Error enhancing audio: {e}")
            return audio_path
    
    def continuous_listen(self, callback, stop_event=None):
        """
        Continuously listen and process speech
        Args:
            callback: Function to call with transcribed text
            stop_event: Threading event to stop listening
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                
                print("Continuous listening started. Speak anytime...")
                
                while True:
                    if stop_event and stop_event.is_set():
                        break
                    
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                        text = self.transcribe_audio(audio)
                        
                        if text and "error" not in text.lower():
                            callback(text)
                            
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error in continuous listening: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error starting continuous listening: {e}")


# Flask route for voice processing
from flask import request, jsonify
from app import app

voice_processor = VoiceProcessor()

@app.route('/api/voice/transcribe', methods=['POST'])
def transcribe_voice():
    """
    API endpoint to transcribe voice input
    Expects audio file in request
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save temporarily
        temp_path = 'temp_audio.wav'
        audio_file.save(temp_path)
        
        # Enhance audio
        enhanced_path = voice_processor.enhance_audio(temp_path)
        
        # Transcribe
        text = voice_processor.process_audio_file(enhanced_path)
        
        # Clean up
        import os
        os.remove(temp_path)
        if enhanced_path != temp_path:
            os.remove(enhanced_path)
        
        return jsonify({
            'success': True,
            'text': text
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/voice/speak', methods=['POST'])
def speak_text():
    """
    API endpoint to convert text to speech
    Returns audio file
    """
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate speech
        output_path = 'temp_speech.mp3'
        voice_processor.engine.save_to_file(text, output_path)
        voice_processor.engine.runAndWait()
        
        # Return audio file
        return send_file(output_path, mimetype='audio/mpeg')
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Example usage
if __name__ == "__main__":
    processor = VoiceProcessor()
    
    print("MediMind Voice Processing Test")
    print("="*50)
    
    # Test text-to-speech
    print("\nTesting Text-to-Speech...")
    processor.speak("Hello! I am MediMind, your AI healthcare assistant. Please describe your symptoms.")
    
    # Test speech recognition
    print("\nTesting Speech Recognition...")
    print("Please speak your symptoms when prompted...")
    
    text = processor.listen_from_microphone(duration=5)
    print(f"\nTranscribed: {text}")
    
    # Respond
    processor.speak(f"I heard: {text}. Let me analyze your symptoms.")