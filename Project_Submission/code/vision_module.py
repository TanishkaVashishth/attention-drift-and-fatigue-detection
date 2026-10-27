import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial import distance as dist

class VisionModule:
    def __init__(self, model_path='face_landmarker.task'):
        # Resolve model path dynamically if not found directly
        if not os.path.exists(model_path):
            dir_path = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(dir_path, model_path)
            if os.path.exists(candidate):
                model_path = candidate
            else:
                for alt_dir in ['deployment', 'code']:
                    alt_cand = os.path.normpath(os.path.join(dir_path, '..', alt_dir, model_path))
                    if os.path.exists(alt_cand):
                        model_path = alt_cand
                        break

        # Initialize the FaceLandmarker Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Landmark indices (Standard 478 face mesh)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [13, 14, 78, 308]
        
        # Thresholds
        self.EAR_THRESHOLD = 0.22
        self.MAR_THRESHOLD = 0.5
        
        # Calibration
        self.baseline_ear = None
        self.baseline_pose = None
        self.is_calibrated = False

    def get_landmarks(self, frame):
        # MediaPipe Tasks requires mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            # We take the first face detected
            return detection_result.face_landmarks[0]
        return None

    def calculate_ear(self, landmarks, indices):
        # landmarks is a list of NormalizedLandmark objects
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append((lm.x, lm.y))
            
        a = dist.euclidean(coords[1], coords[5])
        b = dist.euclidean(coords[2], coords[4])
        c = dist.euclidean(coords[0], coords[3])
        
        ear = (a + b) / (2.0 * c)
        return ear

    def calculate_mar(self, landmarks):
        top = landmarks[13]
        bottom = landmarks[14]
        left = landmarks[78]
        right = landmarks[308]
        
        vertical = dist.euclidean((top.x, top.y), (bottom.x, bottom.y))
        horizontal = dist.euclidean((left.x, left.y), (right.x, right.y))
        
        mar = vertical / horizontal
        return mar

    def estimate_head_pose(self, landmarks, img_w, img_h):
        # Simplified pose estimation
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        d_left = dist.euclidean((nose.x, nose.y), (left_eye.x, left_eye.y))
        d_right = dist.euclidean((nose.x, nose.y), (right_eye.x, right_eye.y))
        yaw_ratio = d_left / (d_right + 1e-6)
        
        forehead = landmarks[10]
        chin = landmarks[152]
        d_up = dist.euclidean((forehead.x, forehead.y), (nose.x, nose.y))
        d_down = dist.euclidean((nose.x, nose.y), (chin.x, chin.y))
        pitch_ratio = d_up / (d_down + 1e-6)
        
        return yaw_ratio, pitch_ratio

    def calibrate(self, ear_values, pose_values):
        if ear_values:
            self.baseline_ear = np.mean(ear_values)
        if pose_values:
            self.baseline_pose = np.mean(pose_values, axis=0)
        self.is_calibrated = True
        print(f"Calibration Complete: Baseline EAR={self.baseline_ear:.3f}")
