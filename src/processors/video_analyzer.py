import cv2
import mediapipe as mp
import numpy as np
import os
from config.settings import FRAME_EXTRACTION_RATE

class VideoAnalyzer:
    def __init__(self, video_path):
        """
        Initialize the VideoAnalyzer with the path to a video file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def analyze_engagement(self, frame):
        """
        Analyze engagement based on head pose (looking at camera).
        Returns True if engaged, False otherwise.
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return False
            
        face_landmarks = results.multi_face_landmarks[0]
        img_h, img_w, _ = frame.shape
        
        # 3D model points (generic)
        face_3d = []
        face_2d = []
        
        # Landmarks for PnP (Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner)
        # Indices: 1, 152, 33, 263, 61, 291
        
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [1, 152, 33, 263, 61, 291]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
                
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])
        
        # Distortion matrix
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        if not success:
            return False
            
        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        
        # Get angles
        angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # angles[0] = pitch, angles[1] = yaw, angles[2] = roll
        x = angles[0] * 360
        y = angles[1] * 360
        
        # Check if looking at camera (thresholds are heuristic)
        # Typically, if x is between -10 and 10, and y is between -10 and 10
        if -10 < x < 10 and -10 < y < 10:
            return True
        else:
            return False

    def analyze_gestures(self, frame):
        """
        Analyze gestures by tracking wrist positions.
        Returns a list of wrist coordinates [(x, y), ...].
        """
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        wrist_positions = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Wrist is landmark 0
                wrist = hand_landmarks.landmark[0]
                wrist_positions.append((wrist.x, wrist.y))
        
        return wrist_positions

    def analyze_emotions(self, frame):
        """
        Placeholder for emotion analysis.
        Returns a random emotion for now.
        """
        # In a real implementation, we would use:
        # result = DeepFace.analyze(frame, actions=['emotion'])
        # return result['dominant_emotion']
        
        import random
        emotions = ['happy', 'neutral', 'surprise', 'sad', 'angry']
        return random.choice(emotions)

    def process_video(self):
        """
        Process the video frame by frame and return aggregated analysis results.
        """
        cap = cv2.VideoCapture(self.video_path)
        
        frame_count = 0
        engaged_frames = 0
        total_processed_frames = 0
        
        all_wrist_positions = []
        emotion_counts = {}
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            
            # Process only every Nth frame
            if frame_count % FRAME_EXTRACTION_RATE != 0:
                continue
                
            total_processed_frames += 1
            
            # 1. Engagement
            if self.analyze_engagement(frame):
                engaged_frames += 1
                
            # 2. Gestures
            wrists = self.analyze_gestures(frame)
            all_wrist_positions.extend(wrists)
            
            # 3. Emotions
            emotion = self.analyze_emotions(frame)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        cap.release()
        
        # Calculate final scores
        if total_processed_frames == 0:
            return {
                "engagement_score": 0.0,
                "gesture_index": 0.0,
                "dominant_emotion": "neutral"
            }
            
        engagement_score = (engaged_frames / total_processed_frames) * 100
        
        # Calculate gesture intensity (variance of wrist movement)
        if not all_wrist_positions:
            gesture_index = 0.0
        else:
            # Calculate variance of x and y coordinates
            wrist_array = np.array(all_wrist_positions)
            # If we have multiple hands, this simple variance might be noisy, 
            # but it gives a rough idea of "amount of movement/spread"
            variance = np.var(wrist_array, axis=0)
            gesture_index = np.mean(variance) * 1000 # Scale up for readability
            
        # Dominant emotion
        if not emotion_counts:
            dominant_emotion = "neutral"
        else:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            
        return {
            "engagement_score": round(engagement_score, 2),
            "gesture_index": round(gesture_index, 2),
            "dominant_emotion": dominant_emotion
        }
