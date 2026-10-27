"""
Launcher for Multimodal AI Attention and Fatigue Monitor (Web Dashboard)
"""
import os
import subprocess
import sys

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(repo_root, "Project_Submission", "deployment", "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find app at {app_path}")
        sys.exit(1)
        
    print("Starting Multimodal AI Monitor Web Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    main()
