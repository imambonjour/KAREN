import cv2
import numpy as np

COLOR_PERSON_RECOGNIZED = (240, 160, 20)
COLOR_PERSON_UNKNOWN = (0, 70, 250)
COLOR_OBJECT = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (40, 40, 40)
COLOR_FACE_FALLBACK = (200, 200, 200)


def draw_corner_rect(img, pt1, pt2, color, thickness=2, line_len=15):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, thickness)


def _face_center_in_bbox(face_bbox, person_bbox):
    fx1, fy1, fx2, fy2 = face_bbox
    fcx = (fx1 + fx2) / 2
    fcy = (fy1 + fy2) / 2
    px1, py1, px2, py2 = person_bbox
    return px1 <= fcx <= px2 and py1 <= fcy <= py2


class Renderer:
    def __init__(self):
        pass

    def draw_overlay(self, frame, active_tracks, track_cache, fps, counts, yolo_detections=None):
        """Render face/person tracks + YOLO co-detections onto the frame."""
        overlay = frame.copy()

        # Stats HUD
        hud_w, hud_h = 220, 140
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "SYSTEM STATUS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detected: {counts['detected']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Unknown: {counts['unknown']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, "[E] Expand samples", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[U] Assign unknown face", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[R] Reload DB  [Q] Quit", (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 255), 1, cv2.LINE_AA)

        # ── Track which face tracks are "covered" by a YOLO person bbox ──
        covered_track_ids = set()

        # ── 1) Draw YOLO detections (if provided) ──
        if yolo_detections:
            for ydet in yolo_detections:
                x1, y1, x2, y2 = ydet['bbox']
                class_id = ydet['class_id']
                class_name = ydet['class_name']

                if class_id == 0:
                    # Person: find matching face track
                    matched_track = None
                    for track in active_tracks:
                        if _face_center_in_bbox(track.bbox, ydet['bbox']):
                            matched_track = track
                            break
                        iou = self._bbox_iou(track.bbox, ydet['bbox'])
                        if iou > 0.05:
                            matched_track = track
                            break

                    if matched_track is not None:
                        covered_track_ids.add(matched_track.track_id)
                        name = track_cache.get(matched_track.track_id) or "Unknown"
                        color = COLOR_PERSON_RECOGNIZED if name != "Unknown" else COLOR_PERSON_UNKNOWN
                        label = f"{name} (ID:{matched_track.track_id})"
                    else:
                        color = COLOR_PERSON_UNKNOWN
                        label = f"person {ydet['confidence']:.2f}"

                    draw_corner_rect(frame, (x1, y1), (x2, y2), color, thickness=3, line_len=18)
                    self._draw_label_tag(frame, label, x1, y1, color)
                else:
                    # Non-person object — draw directly
                    draw_corner_rect(frame, (x1, y1), (x2, y2), COLOR_OBJECT, thickness=2, line_len=12)
                    label = f"{class_name} {ydet['confidence']:.2f}"
                    self._draw_label_tag(frame, label, x1, y1, COLOR_OBJECT)

        # ── 2) Draw unmatched face tracks (fallback) ──
        for track in active_tracks:
            if track.track_id in covered_track_ids:
                continue

            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cache_val = track_cache.get(track.track_id)
            if cache_val and cache_val != "Unknown":
                name = cache_val
                color = COLOR_PERSON_RECOGNIZED
            else:
                name = "Unknown"
                color = COLOR_PERSON_UNKNOWN

            draw_corner_rect(frame, (x1, y1), (x2, y2), color, thickness=3, line_len=18)
            label = f"{name} (ID:{track.track_id})"
            self._draw_label_tag(frame, label, x1, y1, color)

        return frame

    def _draw_label_tag(self, frame, label, x1, y1, color):
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lbl_y1 = max(0, y1 - label_h - 10)
        lbl_y2 = y1
        cv2.rectangle(frame, (x1, lbl_y1), (x1 + label_w + 10, lbl_y2), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    @staticmethod
    def _bbox_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        u = a1 + a2 - inter
        return inter / u if u > 0 else 0.0
