import cv2
import time
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    RECOGNIZER_THRESHOLD
)
from app.database.db import init_db, get_all_persons, add_person, add_person_batch
from app.detection.face_detector import FaceDetector
from app.tracking.bytetrack import SimpleTracker
from app.recognition.face_recognizer import FaceRecognizer
from app.cache.track_cache import TrackCache
from app.overlay.renderer import Renderer
from app.utils.similarity import find_best_match


def expand_person_samples(cap, detector, recognizer, name, num_samples=10):
    """
    Interactive sample collection session.
    Captures additional face samples and merges them into the existing DB entry.
    Press SPACE to capture, Q to cancel.
    Returns list of new embeddings collected.
    """
    print(f"\n--- EXPAND MODE: Adding more samples for '{name}' ---")
    print(f"Press [SPACE] to capture (need {num_samples}), [Q] to cancel early and save.")
    
    embeddings = []
    sample_count = 0
    window_title = f"EXPAND: {name} — SPACE=capture, Q=done [{sample_count}/{num_samples}]"
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
            
        display = frame.copy()
        dets = detector.detect(frame)
        
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 100), 2)
        
        # HUD
        cv2.rectangle(display, (0, 0), (display.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(display, f"EXPAND: {name} — Captured: {sample_count}/{num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(display, "SPACE=capture  Q=done/save",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        
        cv2.imshow(window_title, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Stopping early, saving collected samples...")
            break
        elif key == ord(' '):
            if len(dets) == 0:
                print("  No face detected! Move into frame.")
            elif len(dets) > 1:
                print("  Multiple faces! Make sure only you're in frame.")
            else:
                det = dets[0]
                aligned = recognizer.align_face(frame, det['landmarks'], bbox=det['bbox'])
                if aligned is not None:
                    emb = recognizer.get_embedding(aligned)
                    embeddings.append(emb)
                    sample_count += 1
                    print(f"  Sample {sample_count}/{num_samples} captured.")
                else:
                    print("  Couldn't crop face. Try again.")
    
    cv2.destroyWindow(window_title)
    return embeddings


def main():
    print("Starting Face Recognition System...")
    print("Keys: [Q] quit  [R] reload DB  [E] expand samples for tracked face  [U] register unknown face")
    
    # Initialize components
    init_db()
    
    print("Loading database persons...")
    db_persons = get_all_persons()
    print(f"Loaded {len(db_persons)} registered identities.")
    
    detector = FaceDetector()
    tracker = SimpleTracker(max_lost_frames=30, iou_threshold=0.3)
    recognizer = FaceRecognizer()
    track_cache = TrackCache()
    renderer = Renderer()
    # Setup Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {CAMERA_INDEX}")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    prev_time = time.time()
    
    # Track the last detections so key presses can access them
    last_detections = []
    last_active_tracks = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
            
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-9)
        prev_time = current_time
        
        # 1. Face Detection
        detections = detector.detect(frame)
        last_detections = detections
        
        # 2. Tracking
        active_tracks = tracker.update(detections)
        last_active_tracks = active_tracks
        
        # 3. Clean stale cache
        # Kirim SEMUA track (termasuk yang masih dalam lost_frames) agar ghost
        # bisa menyimpan bbox terakhir sebelum track benar-benar dihapus tracker.
        active_track_ids = [t.track_id for t in active_tracks]
        all_tracker_tracks = tracker.tracks  # includes lost-but-not-yet-deleted
        track_cache.clear_stale(active_track_ids, active_tracks_with_bbox=all_tracker_tracks)
        
        # 4. Identity matching & caching
        unknown_count = 0
        for track in active_tracks:
            track_id = track.track_id
            name = track_cache.get(track_id)
            
            if name is None:
                # New track: cek ghost cache dulu (wajah yang baru muncul kembali)
                ghost_name = track_cache.lookup_ghost(
                    track.bbox, iou_fn=get_iou_bbox, iou_threshold=0.3
                )
                if ghost_name is not None:
                    track_cache.set(track_id, ghost_name)
                    print(f"Track {track_id} → '{ghost_name}' (recovered from ghost cache)")
                    continue

                # Tidak ada ghost yang cocok → jalankan recognition
                best_iou = -1
                matched_det = None
                for det in detections:
                    iou = get_iou_bbox(track.bbox, det['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        matched_det = det
                
                if matched_det is not None and best_iou > 0.5:
                    aligned_face = recognizer.align_face(frame, matched_det['landmarks'], bbox=matched_det['bbox'])
                    if aligned_face is not None:
                        emb = recognizer.get_embedding(aligned_face)
                        matched_name, sim = find_best_match(emb, db_persons, threshold=RECOGNIZER_THRESHOLD)
                        track_cache.set(track_id, matched_name)
                        print(f"Track {track_id} → '{matched_name}' (sim: {sim:.3f})")
                    else:
                        track_cache.set(track_id, "Unknown")
                else:
                    track_cache.set(track_id, "Unknown")
            
            if track_cache.get(track_id) == "Unknown":
                unknown_count += 1
                
        # 5. Stats
        counts = {"detected": len(active_tracks), "unknown": unknown_count}
        
        # 6. Rendering
        rendered_frame = renderer.draw_overlay(frame, active_tracks, track_cache, fps, counts)
        cv2.imshow("Real-Time Face Recognition", rendered_frame)
        
        # Key handlers
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            print("Reloading database...")
            db_persons = get_all_persons()
            print(f"Reloaded {len(db_persons)} identities.")
        
        elif key == ord('e'):
            # ── EXPAND: add more samples for the first recognized (non-Unknown) track ──
            target_name = None
            for track in last_active_tracks:
                cached = track_cache.get(track.track_id)
                if cached and cached != "Unknown":
                    target_name = cached
                    break
            
            if target_name is None:
                print("[E] No recognized face on screen. Show your face first.")
            else:
                new_embs = expand_person_samples(cap, detector, recognizer, target_name, num_samples=10)
                if new_embs:
                    # Simpan tiap embedding individual — bukan mean!
                    add_person_batch(target_name, new_embs)
                    # Reload DB and reset cache so re-recognition fires immediately
                    db_persons = get_all_persons()
                    track_cache.cache.clear()
                    print(f"[E] Database expanded for '{target_name}'. Reload complete.")
                else:
                    print("[E] No samples collected.")
        
        elif key == ord('u'):
            # ── ASSIGN UNKNOWN: assign an Unknown face to an existing DB person ──
            # Find the first Unknown track that has a matched detection
            unknown_track = None
            unknown_det = None
            for track in last_active_tracks:
                if track_cache.get(track.track_id) == "Unknown":
                    # Find the best matching detection for this track
                    best_iou = -1
                    best_det = None
                    for det in last_detections:
                        iou = get_iou_bbox(track.bbox, det['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_det = det
                    if best_det is not None and best_iou > 0.3:
                        unknown_track = track
                        unknown_det = best_det
                        break

            if unknown_track is None:
                print("[U] No Unknown face found on screen right now.")
            elif not db_persons:
                print("[U] Database is empty. Use the registration script to add someone first.")
            else:
                # Show numbered list of existing names in the terminal
                print("\n[U] Assign this Unknown face to an existing person:")
                for idx, p in enumerate(db_persons):
                    print(f"    [{idx}] {p['name']}")
                print("    [n] Register as NEW person")
                print("    [c] Cancel")
                choice = input("  Enter number / n / c: ").strip().lower()

                if choice == 'c':
                    print("[U] Cancelled.")
                elif choice == 'n':
                    new_name = input("  Enter new person name: ").strip()
                    if new_name:
                        aligned = recognizer.align_face(frame, unknown_det['landmarks'], bbox=unknown_det['bbox'])
                        if aligned is not None:
                            emb = recognizer.get_embedding(aligned)
                            add_person(new_name, emb, num_samples=1)
                            db_persons = get_all_persons()
                            track_cache.set(unknown_track.track_id, new_name)
                            print(f"[U] Registered as new person '{new_name}'. Recognized immediately.")
                        else:
                            print("[U] Could not crop the face. Try again.")
                    else:
                        print("[U] Empty name, cancelled.")
                else:
                    try:
                        chosen_idx = int(choice)
                        if 0 <= chosen_idx < len(db_persons):
                            target_name = db_persons[chosen_idx]['name']
                            # Extract embedding from the current Unknown face crop
                            aligned = recognizer.align_face(frame, unknown_det['landmarks'], bbox=unknown_det['bbox'])
                            if aligned is not None:
                                emb = recognizer.get_embedding(aligned)
                                add_person(target_name, emb, num_samples=1)
                                db_persons = get_all_persons()
                                # Immediately update the cache so the label flips right away
                                track_cache.set(unknown_track.track_id, target_name)
                                print(f"[U] Assigned Unknown (Track {unknown_track.track_id}) → '{target_name}'. DB updated.")
                            else:
                                print("[U] Could not crop the Unknown face. Try again.")
                        else:
                            print("[U] Invalid number.")
                    except ValueError:
                        print("[U] Invalid input, cancelled.")
            
    cap.release()
    cv2.destroyAllWindows()
    print("System stopped.")


def get_iou_bbox(box1, box2):
    """Computes IoU between two bounding boxes in [x1, y1, x2, y2] format."""
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


if __name__ == "__main__":
    main()
