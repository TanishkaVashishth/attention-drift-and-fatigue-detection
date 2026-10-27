import time
import numpy as np
import joblib
import os

class ScoringEngine:
    def __init__(self, vision_module, behavioral_module):
        self.vision = vision_module
        self.behavior = behavioral_module
        self.last_switch_count = 0
        
        # Load ML Model
        self.model_path = 'fatigue_model.pkl'
        if not os.path.exists(self.model_path):
            dir_path = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(dir_path, self.model_path)
            if os.path.exists(candidate):
                self.model_path = candidate
            else:
                for alt_dir in ['deployment', 'code']:
                    alt_cand = os.path.normpath(os.path.join(dir_path, '..', alt_dir, self.model_path))
                    if os.path.exists(alt_cand):
                        self.model_path = alt_cand
                        break

        self.model = None
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"ML Model loaded from {self.model_path}")
        else:
            print("Warning: Fatigue ML model not found. Falling back to rule-based.")

        # State
        self.attention_score = 1.0
        self.fatigue_score = 0.0
        
        # Buffers for smoothing
        self.ear_history = []
        self.pose_history = []
        self.window_size = 20 # Smoothing over ~1 second
        
        # Scaling and recovery
        self.fatigue_recovery_rate = 0.005
        self.attention_recovery_rate = 0.01

    def calculate_scores(self, ear, mar, pose):
        # 1. Fatigue Scoring (ML Based)
        if self.model:
            # Predict probability of class 1 (drowsy)
            features = np.array([[ear, mar]])
            probs = self.model.predict_proba(features)[0]
            # Probabilities: [not_drowsy, drowsy]
            ml_fatigue_prob = probs[1]
            
            # Smooth the ML output
            self.fatigue_score = 0.7 * self.fatigue_score + 0.3 * ml_fatigue_prob
        else:
            # Fallback Rule-based logic
            threshold = self.vision.baseline_ear * 0.85 if self.vision.is_calibrated else 0.22
            if ear < threshold:
                self.fatigue_score = min(1.0, self.fatigue_score + 0.05)
            else:
                self.fatigue_score = max(0.0, self.fatigue_score - self.fatigue_recovery_rate)
            
            if mar > 0.45:
                self.fatigue_score = min(1.0, self.fatigue_score + 0.1)

        # 2. Attention Scoring (Vision-based + Behavioral)
        yaw, pitch = pose
        target_attention = 1.0
        
        # If looking away (Vision)
        if abs(1.0 - yaw) > 0.3 or abs(1.0 - pitch) > 0.3:
            target_attention = 0.2
            
        # If Tab Switch detected (Behavioral)
        current_switches = self.behavior.tab_switch_count
        if current_switches > self.last_switch_count:
            self.attention_score = 0.0 # Instant distraction on switch
            self.last_switch_count = current_switches
            
        # Smooth attention transition
        if self.attention_score > target_attention:
            self.attention_score -= 0.1 # Drop quickly if looking away
        else:
            self.attention_score = min(target_attention, self.attention_score + self.attention_recovery_rate)
            
        # 3. Final Cleanup
        self.fatigue_score = np.clip(self.fatigue_score, 0.0, 1.0)
        self.attention_score = np.clip(self.attention_score, 0.0, 1.0)
        
        return self.attention_score, self.fatigue_score

    def get_status(self):
        if self.fatigue_score > 0.65:
            return "FATIGUED", (0, 0, 255) # Red
        if self.attention_score < 0.5:
            return "DISTRACTED", (0, 165, 255) # Orange
        return "ATTENTIVE", (0, 255, 0) # Green
