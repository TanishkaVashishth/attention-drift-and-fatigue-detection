import cv2
import os
import glob
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from vision_module import VisionModule
import random

def train_model(dataset_path, checkpoint_path='features_cache.joblib'):
    vision = VisionModule()
    
    X = []
    y = []
    processed_files = set()
    
    # 1. Load Checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading cached features from {checkpoint_path}...")
        data = joblib.load(checkpoint_path)
        X = data['X']
        y = data['y']
        processed_files = data.get('files', set())
        print(f"Resuming with {len(X)} samples already processed.")
    
    classes = {'notdrowsy': 0, 'drowsy': 1}
    
    print(f"Scanning Dataset at {dataset_path}...")
    
    for cls_name, label in classes.items():
        path = os.path.join(dataset_path, cls_name)
        all_images = glob.glob(os.path.join(path, "*.jpg"))
        
        # Filter out already processed files
        new_images = [img for img in all_images if img not in processed_files]
        
        print(f"Processing {len(new_images)} new images for {cls_name} (Already had {len([f for f in processed_files if cls_name in f])})...")
        
        count = 0
        for i, img_path in enumerate(new_images):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            landmarks = vision.get_landmarks(img)
            if landmarks:
                ear_l = vision.calculate_ear(landmarks, vision.LEFT_EYE)
                ear_r = vision.calculate_ear(landmarks, vision.RIGHT_EYE)
                ear = (ear_l + ear_r) / 2.0
                mar = vision.calculate_mar(landmarks)
                
                X.append([ear, mar])
                y.append(label)
                processed_files.add(img_path)
                count += 1
            
            # Periodically save checkpoint and print progress
            if (i + 1) % 500 == 0:
                print(f"  > {cls_name}: {i+1}/{len(new_images)} processed...")
                joblib.dump({'X': X, 'y': y, 'files': processed_files}, checkpoint_path)
        
        print(f"Finished {cls_name}. Total samples for this class: {len([f for f in y if f == label])}")
        # Save after class completion
        joblib.dump({'X': X, 'y': y, 'files': processed_files}, checkpoint_path)
        
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("Error: No features extracted.")
        return
        
    print(f"\nFinal Dataset Size: {len(X)} samples.")
    print(f"Training MLP Neural Network (Deep Learning)...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLP Neural Network: 2 hidden layers (64, 32), ReLU activation, Adam solver
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        solver='adam', 
        max_iter=1000, 
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- TRAINING COMPLETE ---")
    print(f"Full Dataset Validation Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save final model
    model_filename = 'fatigue_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Final Model saved as {model_filename}")
    
    return model

if __name__ == "__main__":
    base_path = r"c:\Users\nikun\Desktop\AI PROJECT 2\nhtuddd2\train_data"
    if os.path.exists(base_path):
        train_model(base_path)
    else:
        print(f"Error: Dataset path {base_path} not found.")
