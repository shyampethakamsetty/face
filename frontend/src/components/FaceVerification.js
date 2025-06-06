import React, { useState, useRef, useEffect } from 'react';
import './FaceVerification.css';

const FaceVerification = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [verificationStatus, setVerificationStatus] = useState({
    turnLeft: false,
    turnRight: false,
    blink: false
  });
  const [stepMessage, setStepMessage] = useState('');
  const [consecutivePasses, setConsecutivePasses] = useState(0);
  const [isVerifying, setIsVerifying] = useState(false);
  const [eyeCloseCount, setEyeCloseCount] = useState(0);
  const [lastEyeCloseTime, setLastEyeCloseTime] = useState(null);
  const [wasEyesClosed, setWasEyesClosed] = useState(false);
  const [eyeClosedDuration, setEyeClosedDuration] = useState(0);
  const [lastEyePosition, setLastEyePosition] = useState(null);
  const [eyeMovementDetected, setEyeMovementDetected] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const steps = [
    { id: 'turn_left', label: 'Turn Left' },
    { id: 'turn_right', label: 'Turn Right' },
    { id: 'blink', label: 'Close Eyes Slowly (3 times)' }
  ];

  useEffect(() => {
    startCamera();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640,
          height: 480
        } 
      });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;

      // Set canvas dimensions when video is loaded
      videoRef.current.onloadedmetadata = () => {
        if (canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }
      };
    } catch (err) {
      setStepMessage('Error accessing camera. Please ensure camera permissions are granted.');
    }
  };

  const drawFaceMesh = (landmarks, connections) => {
    if (!landmarks || !connections || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];
      if (startPoint && endPoint) {
        ctx.beginPath();
        ctx.moveTo(startPoint[0] * canvas.width, startPoint[1] * canvas.height);
        ctx.lineTo(endPoint[0] * canvas.width, endPoint[1] * canvas.height);
        ctx.stroke();
      }
    });

    // Draw landmarks
    ctx.fillStyle = '#ffff00';
    landmarks.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0] * canvas.width, point[1] * canvas.height, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const checkLighting = async () => {
    try {
      const response = await fetch('http://localhost:5000/verify-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          image: getBase64Image(),
          expected_node: 'turn_left'
        })
      });
      const data = await response.json();
      
      if (data.success && data.lighting === 'good') {
        setVerificationStatus(prev => ({
          ...prev,
          turnLeft: true
        }));
        setStepMessage('');
      } else {
        setStepMessage(data.message || 'Please move to better lighting to continue');
      }

      if (data.landmarks) {
        drawFaceMesh(data.landmarks, data.connections);
      }
    } catch (err) {
      setStepMessage('Error checking lighting conditions');
    }
  };

  const getBase64Image = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    return canvas.toDataURL('image/jpeg');
  };

  const verifyStep = async () => {
    if (!videoRef.current) return;
    
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
          expected_node: currentStepData.id
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        // Draw face mesh
        if (data.landmarks && data.connections) {
          drawFaceMesh(data.landmarks, data.connections);
        }

        if (data.lighting === 'poor') {
          setStepMessage('Please improve lighting conditions');
          return;
        }

        if (currentStepData.id === 'blink') {
          const now = Date.now();
          
          // Check for eye movement
          if (data.eye_state && data.eye_state.left_eye_center && data.eye_state.right_eye_center) {
            const currentEyePosition = {
              left: data.eye_state.left_eye_center,
              right: data.eye_state.right_eye_center
            };

            if (lastEyePosition) {
              const leftEyeMovement = Math.hypot(
                currentEyePosition.left[0] - lastEyePosition.left[0],
                currentEyePosition.left[1] - lastEyePosition.left[1]
              );
              const rightEyeMovement = Math.hypot(
                currentEyePosition.right[0] - lastEyePosition.right[0],
                currentEyePosition.right[1] - lastEyePosition.right[1]
              );

              if (leftEyeMovement > 0.01 || rightEyeMovement > 0.01) {
                setEyeMovementDetected(true);
                setStepMessage('Eye movement detected - keep your eyes still');
              }
            }
            setLastEyePosition(currentEyePosition);
          }

          if (data.eye_state.is_closed) {
            if (!wasEyesClosed) {
              setWasEyesClosed(true);
              setLastEyeCloseTime(now);
              setStepMessage('Eyes closed - keep them closed');
            } else {
              const duration = (now - lastEyeCloseTime) / 1000;
              setEyeClosedDuration(duration);
              
              if (duration >= 0.5) { // Reduced from 1.0 to 0.5 seconds
                setEyeCloseCount(prev => prev + 1);
                setWasEyesClosed(false);
                setEyeClosedDuration(0);
                setStepMessage(`Eye close ${eyeCloseCount + 1}/3 completed`);
                
                if (eyeCloseCount + 1 >= 3) {
                  setVerificationStatus(prev => ({
                    ...prev,
                    [currentStepData.id]: 'passed'
                  }));
                  
                  const nextStepIndex = currentStep + 1;
                  if (nextStepIndex < steps.length) {
                    setCurrentStep(nextStepIndex);
                    setStepMessage(`Moving to next step: ${steps[nextStepIndex].label}`);
                  } else {
                    setStepMessage('Verification completed!');
                    stopVerification();
                  }
                }
              }
            }
          } else if (wasEyesClosed) {
            setWasEyesClosed(false);
            setEyeClosedDuration(0);
            setStepMessage('Please close your eyes slowly');
          }
        } else {
          if (data.node_passed) {
            setVerificationStatus(prev => ({
              ...prev,
              [currentStepData.id]: 'passed'
            }));
            
            const nextStepIndex = currentStep + 1;
            if (nextStepIndex < steps.length) {
              setCurrentStep(nextStepIndex);
              setStepMessage(`Moving to next step: ${steps[nextStepIndex].label}`);
            } else {
              setStepMessage('Verification completed!');
              stopVerification();
            }
          } else {
            setStepMessage(data.message);
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setStepMessage('Error during verification');
    }
  };

  useEffect(() => {
    let interval;
    if (isVerifying) {
      interval = setInterval(verifyStep, 100);
    }
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isVerifying, currentStep, consecutivePasses]);

  const startVerification = () => {
    setCurrentStep(0);
    setVerificationStatus({
      turnLeft: false,
      turnRight: false,
      blink: false
    });
    setConsecutivePasses(0);
    setEyeCloseCount(0);
    setLastEyeCloseTime(null);
    setWasEyesClosed(false);
    setEyeClosedDuration(0);
    setLastEyePosition(null);
    setEyeMovementDetected(false);
    setIsVerifying(true);
    setStepMessage('Starting verification...');
  };

  const stopVerification = () => {
    setIsVerifying(false);
    setStepMessage('Verification stopped');
  };

  return (
    <div className="face-verification">
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ width: '640px', height: '480px' }}
        />
        <canvas
          ref={canvasRef}
          className="face-mesh-overlay"
          style={{ width: '640px', height: '480px' }}
        />
      </div>

      <div className="verification-status">
        <div className="steps">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`step ${index === currentStep ? 'active' : ''} ${
                verificationStatus[step.id] ? 'completed' : ''
              }`}
            >
              <div className="step-number">{index + 1}</div>
              <div className="step-label">{step.label}</div>
            </div>
          ))}
        </div>
        <div className="message">{stepMessage}</div>
        {!isVerifying && (
          <button className="start-button" onClick={startVerification}>
            Start Verification
          </button>
        )}
      </div>
    </div>
  );
};

export default FaceVerification; 