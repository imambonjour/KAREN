import cv2
import numpy as np
import onnxruntime as ort
from app.config import RECOGNIZER_MODEL_CPU

# Target points for face alignment (InsightFace standard)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose Point
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041]   # Right Mouth Corner
], dtype=np.float32)

class FaceRecognizer:
    def __init__(self, model_path=None):
        self.model_path = model_path or str(RECOGNIZER_MODEL_CPU)
        
        # Load ONNX model
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 112, 112]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def align_face(self, frame, landmarks, bbox=None):
        """Aligns the face using affine transformation based on landmarks. Falls back to crop if landmarks are missing."""
        try:
            if landmarks is not None:
                # Estimate similarity transform
                M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), REFERENCE_LANDMARKS)
                if M is not None:
                    aligned = cv2.warpAffine(frame, M, (self.input_width, self.input_height))
                    return aligned
            
            # Fallback to simple crop
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bw = x2 - x1
                bh = y2 - y1
                # Add 18% margin to include more facial features (hair/ears) like standard ArcFace training samples
                margin_w = int(bw * 0.18)
                margin_h = int(bh * 0.18)
                
                h, w = frame.shape[:2]
                x1_pad = max(0, int(x1 - margin_w))
                y1_pad = max(0, int(y1 - margin_h))
                x2_pad = min(w, int(x2 + margin_w))
                y2_pad = min(h, int(y2 + margin_h))
                
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if crop.size > 0:
                    aligned = cv2.resize(crop, (self.input_width, self.input_height))
                    return aligned
            return None
        except Exception as e:
            print(f"Alignment/Crop error: {e}")
            return None

    def get_embedding(self, face_img):
        """
        Extracts embedding vector from a cropped or aligned BGR face image.
        Returns:
            np.ndarray: 512-dimensional embedding vector (L2-normalized)
        """
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize if not already matching the input shape
        if rgb_img.shape[0] != self.input_height or rgb_img.shape[1] != self.input_width:
            rgb_img = cv2.resize(rgb_img, (self.input_width, self.input_height))
            
        # Normalization: (x - 127.5) / 127.5
        blob = (rgb_img.astype(np.float32) - 127.5) / 127.5
        
        # Transpose to channels-first [1, 3, 112, 112]
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # Inference
        outputs = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0][0]  # Shape (512,)
        
        # L2 Normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
