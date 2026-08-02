import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.config import CAMERA_INDEX
from app.database.db import init_db, add_person
from app.detection.face_detector import FaceDetector
from app.recognition.face_recognizer import FaceRecognizer

def register_from_camera(name, num_samples=10):
    print(f"Initializing camera for registering: {name}")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Initialize DB
    init_db()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
        
    cv2.namedWindow("Register Person - Press SPACE to Capture, Q to Quit", cv2.WINDOW_NORMAL)
    
    embeddings = []
    sample_count = 0
    
    print("\nInstructions:")
    print("1. Look directly at the camera.")
    print("2. Tilt your head slightly to capture different angles (left, right, up, down).")
    print("3. Press [SPACE] to capture a face sample.")
    print("4. Press [Q] to quit/cancel registration.")
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        display_frame = frame.copy()
        
        # Detect faces to show visual feedback
        dets = detector.detect(frame)
        
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            # Draw temporary blue box indicating detected face
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        # Draw status text
        cv2.putText(
            display_frame,
            f"Samples Captured: {sample_count}/{num_samples}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        cv2.imshow("Register Person - Press SPACE to Capture, Q to Quit", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Registration cancelled by user.")
            break
        elif key == ord(' '):
            # Capture sample
            if len(dets) == 0:
                print("No face detected! Try aligning your face with the camera.")
            elif len(dets) > 1:
                print("Multiple faces detected! Make sure only you are in the frame.")
            else:
                det = dets[0]
                # Align and get embedding
                aligned_face = recognizer.align_face(frame, det['landmarks'], bbox=det['bbox'])
                if aligned_face is not None:
                    emb = recognizer.get_embedding(aligned_face)
                    embeddings.append(emb)
                    sample_count += 1
                    print(f"Sample {sample_count}/{num_samples} captured successfully.")
                else:
                    print("Face alignment failed. Please look straight at the camera.")
                    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(embeddings) > 0:
        # Calculate mean embedding
        mean_emb = np.mean(embeddings, axis=0)
        # Normalize
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        
        # Save to DB — pass num_samples=len(embeddings) so DB can weight-average
        success = add_person(name, mean_emb, num_samples=len(embeddings))
        if success:
            return True
        else:
            print("\nFailed to save to database.")
            return False
    else:
        print("\nNo samples captured. Registration failed.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a person into the face recognition system database.")
    parser.add_argument("--name", type=str, required=True, help="Name of the person to register.")
    parser.add_argument("--samples", type=int, default=5, help="Number of face samples to average.")
    args = parser.parse_args()
    
    register_from_camera(args.name, args.samples)
