from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constants for verification
TURN_THRESHOLD = 30  # Increased threshold for more pronounced turns
BRIGHTNESS_THRESHOLD = 50
CONTRAST_THRESHOLD = 20
EAR_THRESHOLD = 0.25  # More sensitive threshold for eyes closed
EAR_OPEN_THRESHOLD = 0.30  # More sensitive threshold for eyes open
MIN_EYE_CLOSED_TIME = 0.5  # Reduced minimum time for eyes closed (in seconds)

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
        
        # Invert yaw angle for mirrored video
        yaw = -float(euler_angles[1, 0])
        
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

def detect_eye_state(landmarks):
    try:
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
        
        # Calculate eye aspect ratios
        left_eye = points[33:46]
        right_eye = points[362:375]
        
        # Calculate EAR for left eye
        left_ear = np.mean([
            np.linalg.norm(left_eye[1] - left_eye[5]),
            np.linalg.norm(left_eye[2] - left_eye[4])
        ]) / (2.0 * np.linalg.norm(left_eye[0] - left_eye[3]))
        
        # Calculate EAR for right eye
        right_ear = np.mean([
            np.linalg.norm(right_eye[1] - right_eye[5]),
            np.linalg.norm(right_eye[2] - right_eye[4])
        ]) / (2.0 * np.linalg.norm(right_eye[0] - right_eye[3]))
        
        # Average EAR
        ear = float((left_ear + right_ear) / 2.0)
        
        # Check if eyes are closed (more sensitive detection)
        is_closed = left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD
        is_open = left_ear > EAR_OPEN_THRESHOLD and right_ear > EAR_OPEN_THRESHOLD
        
        # Calculate eye movement
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        return {
            'is_closed': bool(is_closed),
            'is_open': bool(is_open),
            'ear': float(ear),
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            'left_eye_center': [float(left_eye_center[0]), float(left_eye_center[1])],
            'right_eye_center': [float(right_eye_center[0]), float(right_eye_center[1])]
        }
    except Exception as e:
        print(f"Error detecting eye state: {str(e)}")
        return {
            'is_closed': False,
            'is_open': False,
            'ear': 1.0,
            'left_ear': 1.0,
            'right_ear': 1.0,
            'left_eye_center': [0.0, 0.0],
            'right_eye_center': [0.0, 0.0]
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
        
        # Check lighting first
        lighting = check_lighting(image)
        if lighting['status'] == 'poor':
            return jsonify({
                'success': True,
                'lighting': 'poor',
                'node_passed': False,
                'message': 'Lighting is too poor',
                'landmarks': landmarks,
                'connections': FACE_MESH_CONNECTIONS
            })
        
        # Initialize variables
        node_passed = False
        message = ''
        head_pose = None
        eye_state = None
        
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
                
        elif expected_node == 'blink':
            eye_state = detect_eye_state(face_landmarks)
            if eye_state['is_closed']:
                message = 'Eyes closed - keep them closed for a moment'
            elif eye_state['is_open']:
                message = 'Eyes open - close them slowly'
            else:
                message = 'Please close your eyes slowly'
        
        # Convert head_pose values to native Python types if present
        if head_pose:
            head_pose = {
                'pitch': float(head_pose['pitch']),
                'yaw': float(head_pose['yaw']),
                'roll': float(head_pose['roll'])
            }
        
        return jsonify({
            'success': True,
            'lighting': 'good',
            'node_passed': bool(node_passed),
            'message': message,
            'landmarks': landmarks,
            'connections': FACE_MESH_CONNECTIONS,
            'head_pose': head_pose if expected_node in ['turn_left', 'turn_right'] else None,
            'eye_state': eye_state if expected_node == 'blink' else None
        })
            
    except Exception as e:
        print(f"Error in verify-frame endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 