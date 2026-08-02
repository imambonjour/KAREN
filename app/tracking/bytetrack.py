import numpy as np

def get_iou(box1, box2):
    """Computes Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union

class Track:
    def __init__(self, track_id, bbox, confidence=0.0):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.lost_frames = 0
        self.is_activated = True

class SimpleTracker:
    def __init__(self, max_lost_frames=30, iou_threshold=0.3):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        Updates the tracker with new detections.
        detections: list of dict, each containing 'bbox' [x1, y1, x2, y2] and 'confidence'
        Returns:
            list of Track: active tracks
        """
        # 1. Separate into box and conf
        det_boxes = [det['bbox'] for det in detections]
        det_confs = [det['confidence'] for det in detections]
        
        # 2. Match existing tracks with new detections using IoU
        matched_detections = set()
        matched_tracks = set()
        
        # Sort existing tracks: active first
        active_tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]
        
        # Compute IoU matrix
        if len(active_tracks) > 0 and len(det_boxes) > 0:
            iou_matrix = np.zeros((len(active_tracks), len(det_boxes)), dtype=np.float32)
            for i, track in enumerate(active_tracks):
                for j, det_box in enumerate(det_boxes):
                    iou_matrix[i, j] = get_iou(track.bbox, det_box)
                    
            # Greedy matching
            for i, track in enumerate(active_tracks):
                # Find best detection for this track
                best_det_idx = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i, best_det_idx]
                
                if best_iou >= self.iou_threshold and best_det_idx not in matched_detections:
                    track.bbox = det_boxes[best_det_idx]
                    track.confidence = det_confs[best_det_idx]
                    track.lost_frames = 0
                    matched_tracks.add(track.track_id)
                    matched_detections.add(best_det_idx)

        # 3. Handle unmatched tracks (increment lost frame count)
        for track in self.tracks:
            if track.track_id not in matched_tracks:
                track.lost_frames += 1

        # 4. Handle unmatched detections (create new tracks)
        for j, det_box in enumerate(det_boxes):
            if j not in matched_detections:
                new_track = Track(self.next_id, det_box, det_confs[j])
                self.tracks.append(new_track)
                self.next_id += 1

        # 5. Clean up old tracks
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]

        # 6. Return only tracks that are currently active (lost_frames == 0)
        return [t for t in self.tracks if t.lost_frames == 0]
