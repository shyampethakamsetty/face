from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import base64
from PIL import Image
import io
import insightface
from insightface.app import FaceAnalysis
import onnxruntime

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize InsightFace
try:
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Successfully loaded InsightFace model")
except Exception as e:
    print(f"Error loading InsightFace model: {str(e)}")
    face_analyzer = None

# Constants for verification
TURN_THRESHOLD = 30  # Increased threshold for more pronounced turns
BRIGHTNESS_THRESHOLD = 50
CONTRAST_THRESHOLD = 20
FACE_DISTANCE_THRESHOLD = 0.3  # Threshold for face distance (normalized)
FACE_CENTER_THRESHOLD = 0.2  # Threshold for face centering (normalized)
FACE_STRAIGHT_THRESHOLD = 2.0  # Threshold for face straightness (degrees)

# Load age and gender models
try:
    age_model = load_model('models/age_model.h5')
    gender_model = load_model('models/gender_model.h5')
except:
    print("Warning: Age and gender models not found. Please ensure models are in the models directory.")

# Face mesh connections for visualization
FACE_MESH_CONNECTIONS = [
    # Face outline
    [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
    [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
    [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
    [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
    [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
    [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
    # Left eye
    [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154],
    [154, 155], [155, 133], [133, 173], [173, 157], [157, 158], [158, 159],
    [159, 160], [160, 161], [161, 246], [246, 33],
    # Right eye
    [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390],
    [390, 249], [249, 263], [263, 466], [466, 388], [388, 387], [387, 386],
    [386, 385], [385, 384], [384, 398], [398, 362]
]

def read_base64_image(base64_string):
    try:
        # Remove the data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL Image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        return None

def calculate_head_pose(landmarks, image_shape):
    try:
        # Convert landmarks to numpy array
        points = np.array([[lm.x * image_shape[1], lm.y * image_shape[0]] for lm in landmarks.landmark])
        
        # Calculate head pose using solvePnP
        face_3d = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [0.0, -330.0, -65.0],     # Chin
            [-225.0, 170.0, -135.0],  # Left eye left corner
            [225.0, 170.0, -135.0],   # Right eye right corner
            [-150.0, -150.0, -125.0], # Left mouth corner
            [150.0, -150.0, -125.0]   # Right mouth corner
        ], dtype=np.float64)
        
        # Get corresponding 2D points
        face_2d = np.array([
            points[1],   # Nose tip
            points[152], # Chin
            points[226], # Left eye left corner
            points[446], # Right eye right corner
            points[57],  # Left mouth corner
            points[287]  # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix
        size = image_shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4,1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            face_3d, face_2d, camera_matrix, dist_coeffs)
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Get Euler angles
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        # Invert yaw angle for mirrored video and correct the direction
        yaw = float(euler_angles[1, 0])
        
        return {
            'pitch': float(euler_angles[0, 0]),
            'yaw': yaw,
            'roll': float(euler_angles[2, 0])
        }
    except Exception as e:
        print(f"Error calculating head pose: {str(e)}")
        return None

def check_lighting(image):
    try:
        # Convert to grayscale for lighting detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness
        brightness = float(np.mean(gray))
        
        # Calculate contrast
        contrast = float(np.std(gray))
        
        is_good_lighting = brightness > BRIGHTNESS_THRESHOLD and contrast > CONTRAST_THRESHOLD
        
        return {
            'status': 'good' if is_good_lighting else 'poor',
            'brightness': brightness,
            'contrast': contrast
        }
    except Exception as e:
        print(f"Error checking lighting: {str(e)}")
        return {'status': 'poor', 'brightness': 0, 'contrast': 0}

def check_face_position(landmarks, image_shape):
    try:
        # Get face bounding box
        points = np.array([[lm.x * image_shape[1], lm.y * image_shape[0]] for lm in landmarks.landmark])
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Calculate face size relative to image
        face_width = (x_max - x_min) / image_shape[1]
        face_height = (y_max - y_min) / image_shape[0]
        face_size = max(face_width, face_height)
        
        # Calculate face center position
        face_center_x = (x_min + x_max) / (2 * image_shape[1])
        face_center_y = (y_min + y_max) / (2 * image_shape[0])
        
        # Check if face is centered
        is_centered = (
            abs(face_center_x - 0.5) < FACE_CENTER_THRESHOLD and
            abs(face_center_y - 0.5) < FACE_CENTER_THRESHOLD
        )
        
        # Check if face is close enough
        is_close = face_size > FACE_DISTANCE_THRESHOLD
        
        return {
            'is_centered': bool(is_centered),
            'is_close': bool(is_close),
            'face_size': float(face_size),
            'face_center': [float(face_center_x), float(face_center_y)]
        }
    except Exception as e:
        print(f"Error checking face position: {str(e)}")
        return {
            'is_centered': False,
            'is_close': False,
            'face_size': 0.0,
            'face_center': [0.5, 0.5]
        }

def analyze_face(image, landmarks):
    try:
        if face_analyzer is None:
            raise Exception("Face analyzer not initialized")

        # Get face region using MediaPipe landmarks
        points = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in landmarks.landmark])
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Add padding
        padding = 0.2
        x_min = max(0, int(x_min - (x_max - x_min) * padding))
        y_min = max(0, int(y_min - (y_max - y_min) * padding))
        x_max = min(image.shape[1], int(x_max + (x_max - x_min) * padding))
        y_max = min(image.shape[0], int(y_max + (y_max - y_min) * padding))
        
        # Extract face region
        face_region = image[y_min:y_max, x_min:x_max]
        
        # Resize to a standard size for better analysis
        face_resized = cv2.resize(face_region, (640, 640))
        
        # Convert to RGB for InsightFace
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Analyze face
        faces = face_analyzer.get(face_rgb)
        
        if not faces:
            # If InsightFace fails, try analyzing the original image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_analyzer.get(image_rgb)
            
            if not faces:
                raise Exception("No face detected in the image")
        
        face = faces[0]  # Get the first face
        
        # Get gender prediction
        gender = 'Male' if face.gender == 1 else 'Female'
        
        # Get gender confidence
        gender_confidence = 0.0
        try:
            # Try to get confidence from various possible attributes
            if hasattr(face, 'gender_score') and face.gender_score is not None:
                gender_confidence = float(face.gender_score)
            elif hasattr(face, 'det_score') and face.det_score is not None:
                gender_confidence = float(face.det_score)
            elif hasattr(face, 'score') and face.score is not None:
                gender_confidence = float(face.score)
            
            # If we still have 0.0, try to get it from the face object's dictionary
            if gender_confidence == 0.0 and hasattr(face, '__dict__'):
                face_dict = face.__dict__
                for key in ['gender_score', 'det_score', 'score']:
                    if key in face_dict and face_dict[key] is not None:
                        gender_confidence = float(face_dict[key])
                        break
            
            # Print debug information
            print("Face object attributes:", dir(face))
            print("Face object dict:", face.__dict__ if hasattr(face, '__dict__') else "No dict")
            print(f"Gender: {gender}, Confidence: {gender_confidence}")
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error getting gender confidence: {str(e)}")
            gender_confidence = 0.0
        
        return {
            'gender': gender,
            'gender_confidence': gender_confidence,
            'face_bbox': {
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max)
            }
        }
    except Exception as e:
        print(f"Error analyzing face: {str(e)}")
        return {
            'gender': 'Unknown',
            'gender_confidence': 0.0,
            'face_bbox': None
        }

@app.route('/verify-frame', methods=['POST'])
def verify_frame():
    try:
        # Get request data
        data = request.get_json()
        if not data or 'image' not in data or 'expected_node' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        # Read and process image
        image = read_base64_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return jsonify({
                'success': True,
                'lighting': 'poor',
                'node_passed': False,
                'message': 'No face detected',
                'landmarks': None
            })
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to list format for visualization
        landmarks = [[float(lm.x), float(lm.y), float(lm.z)] for lm in face_landmarks.landmark]
        
        # Initialize variables
        node_passed = False
        message = ''
        head_pose = None
        
        # Verify the expected node
        expected_node = data['expected_node']
        
        if expected_node == 'turn_left':
            head_pose = calculate_head_pose(face_landmarks, image.shape)
            if head_pose:
                yaw = head_pose['yaw']
                if yaw < -TURN_THRESHOLD:
                    node_passed = True
                    message = 'Turn left detected'
                else:
                    message = f'Please turn your head more to the left (current angle: {yaw:.1f}°)'
            else:
                message = 'Could not detect head pose'
                
        elif expected_node == 'turn_right':
            head_pose = calculate_head_pose(face_landmarks, image.shape)
            if head_pose:
                yaw = head_pose['yaw']
                if yaw > TURN_THRESHOLD:
                    node_passed = True
                    message = 'Turn right detected'
                else:
                    message = f'Please turn your head more to the right (current angle: {yaw:.1f}°)'
            else:
                message = 'Could not detect head pose'
                
        elif expected_node == 'analyze_face':
            # Check all conditions simultaneously
            conditions = {
                'lighting': False,
                'angle': False,
                'position': False
            }
            messages = []
            
            # Check lighting
            lighting = check_lighting(image)
            conditions['lighting'] = lighting['status'] == 'good'
            if not conditions['lighting']:
                messages.append('Please improve lighting conditions')
            
            # Check face angle
            head_pose = calculate_head_pose(face_landmarks, image.shape)
            if head_pose:
                yaw = head_pose['yaw']
                conditions['angle'] = abs(yaw) <= 10  # Using -10 to 10 degrees range
                if not conditions['angle']:
                    messages.append(f'Please face straight ahead (current angle: {yaw:.1f}°)')
            
            # Check face position
            face_position = check_face_position(face_landmarks, image.shape)
            conditions['position'] = face_position['is_close'] and face_position['is_centered']
            if not face_position['is_close']:
                messages.append('Please move your face closer to the camera')
            elif not face_position['is_centered']:
                messages.append('Please center your face in the frame')
            
            # If all conditions are met
            if all(conditions.values()):
                if 'snapshot_taken' in data and data['snapshot_taken']:
                    analysis = analyze_face(image, face_landmarks)
                    if analysis['gender'] != 'Unknown':  # Check if analysis was successful
                        node_passed = True
                        message = f'Analysis complete: {analysis["gender"]}, {analysis["gender_confidence"]}'
                    else:
                        message = 'Could not analyze face. Please try again.'
                else:
                    message = 'Hold still for analysis...'
                    node_passed = True
            else:
                message = ' | '.join(messages)
            
            # Prepare response
            response = {
                'success': True,
                'lighting': 'good' if conditions['lighting'] else 'poor',
                'node_passed': bool(node_passed),
                'message': message,
                'landmarks': landmarks,
                'connections': FACE_MESH_CONNECTIONS,
                'head_pose': head_pose,
                'face_position': face_position,
                'conditions': conditions
            }
            
            # Add analysis results if available
            if 'snapshot_taken' in data and data['snapshot_taken'] and node_passed:
                response['analysis'] = analyze_face(image, face_landmarks)
            
            return jsonify(response)
        
        # For other nodes (turn_left, turn_right)
        response = {
            'success': True,
            'lighting': 'good',
            'node_passed': bool(node_passed),
            'message': message,
            'landmarks': landmarks,
            'connections': FACE_MESH_CONNECTIONS,
            'head_pose': head_pose if expected_node in ['turn_left', 'turn_right'] else None
        }
        
        return jsonify(response)
            
    except Exception as e:
        print(f"Error in verify-frame endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 