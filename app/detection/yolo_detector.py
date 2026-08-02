from ultralytics import YOLO
from app.config import YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD


class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=None):
        self.model_path = model_path or str(YOLO_MODEL_PATH)
        self.conf_threshold = conf_threshold or YOLO_CONF_THRESHOLD
        self.model = YOLO(self.model_path)

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': results.names[int(box.cls[0])],
            }
            detections.append(detection)
        return detections
