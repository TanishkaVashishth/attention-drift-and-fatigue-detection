import cv2
import os
import glob
import numpy as np
from vision_module import VisionModule

def validate_dataset(dataset_path):
    vision = VisionModule()
    
    classes = ['drowsy', 'notdrowsy']
    results = {'drowsy': [], 'notdrowsy': []}
    
    print(f"Validating thresholds on NHTU Dataset at: {dataset_path}")
    
    for cls in classes:
        path = os.path.join(dataset_path, cls)
        # Get first 50 images for quick validation
        images = glob.glob(os.path.join(path, "*.jpg"))[:50]
        
        print(f"Processing class: {cls} ({len(images)} images)...")
        
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            landmarks = vision.get_landmarks(img)
            if landmarks:
                ear_l = vision.calculate_ear(landmarks, vision.LEFT_EYE)
                ear_r = vision.calculate_ear(landmarks, vision.RIGHT_EYE)
                ear = (ear_l + ear_r) / 2.0
                mar = vision.calculate_mar(landmarks)
                
                results[cls].append((ear, mar))
    
    # Analyze results
    for cls in classes:
        data = np.array(results[cls])
        if len(data) > 0:
            avg_ear = np.mean(data[:, 0])
            avg_mar = np.mean(data[:, 1])
            print(f"\nResults for {cls.upper()}:")
            print(f"  Avg EAR: {avg_ear:.3f}")
            print(f"  Avg MAR: {avg_mar:.3f}")
        else:
            print(f"No results for {cls}")

    # SUGGESTED THRESHOLDS
    drowsy_ear = np.mean([x[0] for x in results['drowsy']]) if results['drowsy'] else 0
    not_drowsy_ear = np.mean([x[0] for x in results['notdrowsy']]) if results['notdrowsy'] else 0
    
    if drowsy_ear and not_drowsy_ear:
        suggested = (drowsy_ear + not_drowsy_ear) / 2.0
        print(f"\n--- THRESHOLD TUNING ---")
        print(f"Optimal EAR Threshold suggested: {suggested:.3f}")
        print(f"Current System EAR Threshold: {vision.EAR_THRESHOLD}")

if __name__ == "__main__":
    base_path = r"c:\Users\nikun\Desktop\AI PROJECT 2\nhtuddd2\train_data"
    validate_dataset(base_path)
