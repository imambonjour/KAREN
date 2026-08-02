import cv2
import numpy as np
import onnxruntime as ort
from app.config import (
    DETECTOR_MODEL_CPU,
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_NMS_THRESHOLD
)

class FaceDetector:
    def __init__(self, model_path=None, conf_threshold=None, nms_threshold=None):
        self.model_path = model_path or str(DETECTOR_MODEL_CPU)
        self.conf_threshold = conf_threshold or DETECTOR_CONF_THRESHOLD
        self.nms_threshold = nms_threshold or DETECTOR_NMS_THRESHOLD
        
        # Load ONNX model
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resizes image to a square shape with padding, keeping the aspect ratio."""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top)

    def detect(self, frame):
        """
        Detects faces in the given BGR frame.
        Returns:
            list of dict: [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'landmarks': np.ndarray (5, 2)
            }]
        """
        h, w = frame.shape[:2]
        
        # Preprocessing
        blob_img, ratio, (pad_w, pad_h) = self._letterbox(frame, (self.input_height, self.input_width))
        
        # Convert BGR to RGB, normalize, transpose to channels-first
        blob = blob_img.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # Inference
        outputs = self.session.run(None, {self.input_name: blob})
        output = outputs[0]  # Shape is usually (1, 15, 8400) or similar
        
        # Post-processing
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dim -> (15, 8400)
            
        # Transpose to (8400, 15)
        output = np.transpose(output, (1, 0))

        boxes = []
        confidences = []
        landmarks_list = []

        # YOLOv8 output format: [cx, cy, nw, nh, confidence, ...]
        num_coords = output.shape[1]
        
        for detection in output:
            confidence = detection[4]
            if confidence >= self.conf_threshold:
                cx, cy, nw, nh = detection[0:4]
                
                # Convert center x,y, w, h to corner x1, y1, w, h (input size coordinates)
                x1 = cx - nw / 2
                y1 = cy - nh / 2
                
                # Rescale boxes to original frame dimensions
                orig_x1 = int((x1 - pad_w) / ratio)
                orig_y1 = int((y1 - pad_h) / ratio)
                orig_nw = int(nw / ratio)
                orig_nh = int(nh / ratio)
                
                # Bounding box constraints
                orig_x1 = max(0, orig_x1)
                orig_y1 = max(0, orig_y1)
                orig_x2 = min(w, orig_x1 + orig_nw)
                orig_y2 = min(h, orig_y1 + orig_nh)
                
                boxes.append([orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1])
                confidences.append(float(confidence))
                
                # Extract landmarks if available in model outputs (e.g. 15 channels)
                if num_coords >= 15:
                    landmarks = []
                    for i in range(5):
                        lx = detection[5 + i * 2]
                        ly = detection[5 + i * 2 + 1]
                        orig_lx = int((lx - pad_w) / ratio)
                        orig_ly = int((ly - pad_h) / ratio)
                        landmarks.append([orig_lx, orig_ly])
                    landmarks_list.append(np.array(landmarks))
                else:
                    landmarks_list.append(None)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                box = boxes[idx]
                x1, y1, bw, bh = box
                results.append({
                    'bbox': [x1, y1, x1 + bw, y1 + bh],
                    'confidence': confidences[idx],
                    'landmarks': landmarks_list[idx]
                })
                
        return results
