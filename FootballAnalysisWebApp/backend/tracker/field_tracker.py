from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np


class FieldTracker:
    def __init__(self, model_path):
        """
        Initialise le modèle de détection des lignes du terrain de football et le tracker.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()  # Utilisation d'un tracker pour les lignes
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8
        )

    def detect_field_lines(self, frames):
        """
        Détecte les lignes du terrain dans une séquence de frames.
        """
        detections = []
        for frame in frames:
            results = self.model.predict(source=frame, conf=0.3)
            for result in results:
                if result.keypoints:
                    keypoints_array = result.keypoints.data.cpu().numpy()  # Convertir en tableau numpy
                    print("Keypoints détectés :", keypoints_array)
                else:
                    print("Aucun keypoint détecté.")
        return detections

    def track_field_lines(self, frames, read_from_strub = False, stub_path=None):
        """
        Suit les lignes du terrain sur les frames et les stocke dans un fichier .pkl si nécessaire.
        """
        if read_from_strub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        tracks = {"field_lines": []}

        # Détecter les lignes du terrain
        detections = self.detect_field_lines(frames)
        tracks['field_lines'].append({})

        for frame_num, key_points in enumerate(detections):
            # Mise à jour des keypoints dans le tracker
            tracked_key_points = self.tracker.update(key_points)
            tracks["field_lines"].append({
                "frame_num": frame_num,
                "key_points": [
                    {"id": kp.id, "position": kp.position} for kp in tracked_key_points
                ]
            })

        # Sauvegarde dans un fichier pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


    def annotate_field_lines(self, video_frames, detections):
        """
        Annote les frames avec les lignes détectées en traçant des cercles rouges aux positions des points clés.
    
        :param video_frames: Liste des frames de la vidéo.
        :param detections: Dictionnaire contenant les lignes de terrain détectées, avec des points clés.
        :return: Liste des frames annotées.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            # Crée une copie de la frame pour ne pas la modifier directement
            frame = frame.copy()

            # Récupère les données des points clés pour la frame actuelle
            field = detections["field_lines"][frame_num]  # Accès direct à la liste par index

            # Dessine un cercle rouge pour chaque point clé
            for kp in field["key_points"]:
                position = kp["position"]  # Exemple : (x, y)
                color = (0, 0, 255)  # Rouge (BGR)
                radius = 5           # Rayon du cercle
                thickness = -1       # -1 pour un cercle plein

                # Trace le cercle
                cv2.circle(frame, position, radius, color, thickness)

            # Ajoute la frame annotée à la liste de sortie
            output_video_frames.append(frame)

        return output_video_frames



