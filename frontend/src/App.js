import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import FaceVerification from './components/FaceVerification';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={
            <div className="home-container">
              <h2>Home</h2>
              <p>Welcome to Face Verification System</p>
              <Link to="/verify" className="verify-link">
                Start Face Verification
              </Link>
            </div>
          } />
          <Route path="/verify" element={
            <div>
              <h2>Face Verification</h2>
              <FaceVerification />
            </div>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
