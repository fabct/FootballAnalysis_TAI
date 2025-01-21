from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
from tqdm import tqdm

class FieldTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8
        )

    def track_field_lines(self, frames, read_from_strub=False, stub_path=None, confidence_threshold=0.5):
        if read_from_strub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        tracks = {"field_lines": []}

        for frame_num, frame in tqdm(enumerate(frames), total=len(frames)):
            results = self.model.predict(frame, conf=0.3)
            
            for result in results:
                if not result.keypoints:
                    continue
                
                # Extraction des données YOLO avec filtrage
                kp_data = result.keypoints.cpu().numpy()
                xy = kp_data.xy[0]  # Shape: (N, 2)
                conf = kp_data.conf[0] if kp_data.conf is not None else np.ones(len(xy))
                
                # Application du filtre de confiance
                mask = conf > confidence_threshold
                filtered_xy = xy[mask]
                filtered_conf = conf[mask]

                if len(filtered_xy) == 0:
                    continue

                # Création des bounding boxes pour le tracking
                boxes = np.array([[x-2, y-2, x+2, y+2] for x, y in filtered_xy])
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=filtered_conf,
                    class_id=np.zeros(len(filtered_xy), dtype=int)
                )

                # Tracking
                tracked_detections = self.tracker.update_with_detections(detections=detections)
                
                # Extraction des points trackés
                tracked_points = [
                    ((xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2) 
                    for xyxy in tracked_detections.xyxy
                ]
                
                # Stockage des résultats
                tracks["field_lines"].append({
                    "frame_num": frame_num,
                    "key_points": [
                        {
                            "id": int(track_id),
                            "position": (int(x), int(y)),
                            "confidence": float(conf)
                        } for (x, y), track_id, conf in zip(
                            tracked_points, 
                            tracked_detections.tracker_id,
                            tracked_detections.confidence
                        )
                    ]
                })

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
        return tracks

    def annotate_field_lines(self, video_frames, detections):
        output_video_frames = []
        detection_map = {
            detection["frame_num"]: detection["key_points"]
            for detection in detections
        }

        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy()
            
            if frame_num in detection_map:
                keypoints = detection_map[frame_num]
                positions = [kp["position"] for kp in keypoints]
                
                if len(positions) > 0:
                    xy = np.array(positions, dtype=np.float32).reshape(1, -1, 2)
                    key_points = sv.KeyPoints(xy=xy)
                    
                    annotated_frame = self.vertex_annotator.annotate(
                        scene=annotated_frame,
                        key_points=key_points
                    )

            output_video_frames.append(annotated_frame)
            
        return output_video_frames