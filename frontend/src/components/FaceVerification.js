import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './FaceVerification.css';

const FaceVerification = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [isVerifying, setIsVerifying] = useState(false);
  const [snapshotTaken, setSnapshotTaken] = useState(false);
  const [holdStill, setHoldStill] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState({
    gender: '',
    gender_confidence: 0
  });
  const [showSuccess, setShowSuccess] = useState(false);
  const [finalFrame, setFinalFrame] = useState(null);
  const [isCompleted, setIsCompleted] = useState(false);
  const [showStepComplete, setShowStepComplete] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const successSoundRef = useRef(new Audio('/success-sound.mp3'));

  const steps = [
    { id: 'turn_left', label: 'Turn your head to the left' },
    { id: 'turn_right', label: 'Turn your head to the right' },
    { id: 'analyze_face', label: 'Center your face for analysis' }
  ];

  const [verificationStatus, setVerificationStatus] = useState({
    turn_left: false,
    turn_right: false,
    analyze_face: false
  });

  const [stepMessage, setStepMessage] = useState('');

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setStepMessage('Error accessing camera. Please ensure camera permissions are granted.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startVerification = () => {
    setCurrentStep(0);
    setVerificationStatus({
      turn_left: false,
      turn_right: false,
      analyze_face: false
    });
    setSnapshotTaken(false);
    setAnalysisResult({
      gender: '',
      gender_confidence: 0
    });
    setFinalFrame(null);
    setIsCompleted(false);
    setIsVerifying(true);
    setStepMessage('Starting verification...');
  };

  const stopVerification = () => {
    setIsVerifying(false);
    setIsCompleted(true);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    stopCamera();
    setShowSuccess(true);
  };

  const getBase64Image = () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0);
    return canvas.toDataURL('image/jpeg');
  };

  const drawFaceMesh = (landmarks, connections) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const video = videoRef.current;
    if (!video) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw all landmarks as dots
    ctx.fillStyle = '#00ff00';
    landmarks.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0] * canvas.width, point[1] * canvas.height, 2, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw connections
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 1;
    connections.forEach(connection => {
      const [start, end] = connection;
      ctx.beginPath();
      ctx.moveTo(landmarks[start][0] * canvas.width, landmarks[start][1] * canvas.height);
      ctx.lineTo(landmarks[end][0] * canvas.width, landmarks[end][1] * canvas.height);
      ctx.stroke();
    });
  };

  const playSuccessSound = () => {
    successSoundRef.current.currentTime = 0;
    successSoundRef.current.play().catch(error => {
      console.log('Error playing sound:', error);
    });
  };

  const handleStepComplete = (message) => {
    setShowStepComplete(true);
    playSuccessSound();
    setStepMessage(message);
    setTimeout(() => {
      setShowStepComplete(false);
    }, 2000);
  };

  const verifyStep = async () => {
    if (!videoRef.current || isCompleted) return;
    
    const currentStepData = steps[currentStep];
    if (!currentStepData) return;

    try {
      const image = getBase64Image();
      const response = await fetch('http://localhost:5000/verify-frame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image,
          expected_node: currentStepData.id,
          snapshot_taken: snapshotTaken
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        if (data.landmarks && data.connections && !isCompleted) {
          drawFaceMesh(data.landmarks, data.connections);
        }

        if (currentStepData.id === 'analyze_face') {
          // Check all conditions at once
          if (data.conditions) {
            const allConditionsMet = 
              data.conditions.lighting && 
              data.conditions.angle && 
              data.conditions.position;

            if (!allConditionsMet) {
              setAnalyzing(false);
              const statusMessages = [];
              if (!data.conditions.lighting) {
                statusMessages.push('Please improve lighting conditions');
              }
              if (!data.conditions.angle) {
                statusMessages.push('Please face straight ahead');
              }
              if (!data.conditions.position) {
                if (data.face_position) {
                  if (!data.face_position.is_close) {
                    statusMessages.push('Please move your face closer');
                  }
                  if (!data.face_position.is_centered) {
                    statusMessages.push('Please center your face');
                  }
                }
              }
              setStepMessage(statusMessages.join(' | '));
              return;
            }

            // If all conditions are met and we haven't started holding still
            if (!holdStill) {
              setHoldStill(true);
              setAnalyzing(true);
              setStepMessage('Hold still, analyzing...');
              // Clear the interval to stop sending requests
              if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
              }
              // Send the final request immediately
              const finalImage = getBase64Image();
              fetch('http://localhost:5000/verify-frame', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  image: finalImage,
                  expected_node: currentStepData.id,
                  snapshot_taken: true
                }),
              })
                .then(res => res.json())
                .then(finalData => {
                  if (finalData.success && finalData.analysis) {
                    // Save the final frame with mesh
                    const canvas = canvasRef.current;
                    if (canvas) {
                      setFinalFrame(canvas.toDataURL('image/jpeg'));
                    }
                    setAnalysisResult({
                      gender: finalData.analysis.gender,
                      gender_confidence: finalData.analysis.gender_confidence
                    });
                    setVerificationStatus(prev => ({
                      ...prev,
                      [currentStepData.id]: true
                    }));
                    handleStepComplete(`Analysis complete: ${finalData.analysis.gender} (${(finalData.analysis.gender_confidence * 100).toFixed(1)}% confidence)`);
                    setAnalyzing(false);
                    stopVerification();
                  }
                });
              return;
            }
          }
        } else if (data.node_passed) {
          setVerificationStatus(prev => ({
            ...prev,
            [currentStepData.id]: true
          }));
          
          const nextStepIndex = currentStep + 1;
          if (nextStepIndex < steps.length) {
            handleStepComplete(`Step completed! Moving to: ${steps[nextStepIndex].label}`);
            setCurrentStep(nextStepIndex);
          } else {
            handleStepComplete('Verification completed!');
            stopVerification();
            return;
          }
        } else {
          setStepMessage(data.message);
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setStepMessage('Error during verification');
    }
  };

  useEffect(() => {
    if (isVerifying && !isCompleted) {
      intervalRef.current = setInterval(verifyStep, 100);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isVerifying, currentStep, snapshotTaken, isCompleted]);

  const handleReturnHome = () => {
    navigate('/');
  };

  return (
    <div className="face-verification">
      <div className="verification-container">
        <div className={`video-container${analyzing ? ' analyzing' : ''}`}>
          {!isCompleted ? (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={showSuccess ? 'hidden' : ''}
              />
              <canvas ref={canvasRef} className={showSuccess ? 'hidden' : ''} />
            </>
          ) : (
            finalFrame && (
              <img 
                src={finalFrame} 
                alt="Final verification frame" 
                className="final-frame"
              />
            )
          )}
        </div>

        <div className="progress-bar">
          {steps.map((step, index) => (
            <div 
              key={step.id} 
              className={`progress-node ${index === currentStep ? 'active' : ''} ${
                verificationStatus[step.id] ? 'completed' : ''
              }`}
            >
              <div className="node-number">{index + 1}</div>
              <div className="node-label">{step.label}</div>
            </div>
          ))}
        </div>

        <div className="controls">
          {!isVerifying ? (
            isCompleted ? (
              <button onClick={handleReturnHome} className="return-home-button">
                Return to Home
              </button>
            ) : (
              <button onClick={startVerification} className="start-button">
                Start Verification
              </button>
            )
          ) : (
            <button onClick={stopVerification} className="stop-button">
              Stop Verification
            </button>
          )}
        </div>
        {!isCompleted && (
          <div className={`message ${showStepComplete ? 'step-complete' : ''}`}>
            {stepMessage}
          </div>
        )}
        {analysisResult.gender && (
          <div className="analysis-result">
            <h3>Analysis Results</h3>
            <p>Gender: {analysisResult.gender}</p>
            <p>Confidence: {(analysisResult.gender_confidence * 100).toFixed(1)}%</p>
          </div>
        )}

        {showSuccess && (
          <div className="success-animation">
            <div className="success-checkmark">
              <div className="check-icon">
                <span className="icon-line line-tip"></span>
                <span className="icon-line line-long"></span>
                <div className="icon-circle"></div>
                <div className="icon-fix"></div>
              </div>
            </div>
            <h2>Verification Complete!</h2>
          </div>
        )}
      </div>
    </div>
  );
};

export default FaceVerification; 