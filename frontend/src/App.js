import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Test backend connection
    fetch('http://localhost:5000/api/welcome')
      .then(response => response.json())
      .then(data => {
        setMessage(data.message);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error connecting to backend:', error);
        setMessage('Backend connection failed. Please ensure the Flask server is running.');
        setLoading(false);
      });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>MediMind - AI Healthcare Assistant</h1>
        {loading ? (
          <p>Loading...</p>
        ) : (
          <p>{message}</p>
        )}
        <div style={{ marginTop: '20px' }}>
          <p>Backend Status: {loading ? 'Checking...' : message.includes('Welcome') ? '✅ Connected' : '❌ Disconnected'}</p>
          <p>Frontend: ✅ Running</p>
        </div>
        <div style={{ marginTop: '30px', textAlign: 'left', maxWidth: '600px' }}>
          <h3>Available Features:</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li>✅ User Registration & Authentication</li>
            <li>✅ Symptom Analysis with AI Model</li>
            <li>✅ Medicine Recommendations</li>
            <li>✅ Nearby Hospitals Finder</li>
            <li>✅ Healthcare Provider Directory</li>
            <li>✅ Voice Input Processing</li>
          </ul>
        </div>
        <p style={{ marginTop: '30px', fontSize: '14px', color: '#888' }}>
          Backend API: http://localhost:5000 | Frontend: http://localhost:3000
        </p>
      </header>
    </div>
  );
}

export default App;
