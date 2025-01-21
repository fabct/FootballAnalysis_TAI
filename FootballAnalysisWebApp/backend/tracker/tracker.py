from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width,get_foot_position
from ball_trajectory import BallTrajectoryRNN

# Class Names from YOLO model
#{0: 'Player', 1: 'GoalKeeper', 2: 'Ball', 3: 'Main Referee', 4: 'Side Referee', 5: 'Staff Member'}

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
        

    def get_object_tracks(self, frames, read_from_strub = False, stub_path = None):

        if read_from_strub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frames_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k,v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goal to players object 
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'GoalKeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['Player']
                # convert side referee to referee object 
                if cls_names[class_id] == 'Side Referee':
                    detection_supervision.class_id[object_ind] = cls_names_inv['Main Referee']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                tracks_id= frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    tracks['players'][frames_num][tracks_id] = {"frame_num":frames_num,"bbox":bbox}

                if cls_id == cls_names_inv['Main Referee']:
                    tracks['referees'][frames_num][tracks_id] = {"frame_num":frames_num,"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['Ball']:
                    tracks['ball'][frames_num][1] = {"frame_num":frames_num,"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_annotations(self, video_frames, tracks, team_ball_control, pauses):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dist = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            # Draw players
            for track_id, player in player_dist.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_ellipse(frame, player['bbox'], (0, 255, 0), track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_rectangle(frame, ball['bbox'], (255, 0, 0), 'ball')

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, pauses)

            

            output_video_frames.append(frame)
        
        return output_video_frames

    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control, pauses):
        # Draw a semi-transparent rectaggle 
        frame_height, frame_width, _ = frame.shape
        rect_x1 = int(frame_width * 0.05)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.35)
        rect_y2 = int(frame_height * 0.95)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)
        text_x = rect_x1 + 20
        text_y = rect_y1 + 40
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (text_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # Vérifier si la frame fait partie d'un arrêt de jeu
        for start, end in pauses:
            if start <= frame_num <= end:
                
                cv2.putText(frame, "Stoppage in play", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                break  # Éviter d'écrire plusieurs fois si la frame appartient à plusieurs pauses

        return frame

    def draw_rectangle(self, frame, bbox, color, name):
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}. Expected a tuple or list of four integers.")
    
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1 - 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
   
        # Extraire les coordonnées des boîtes englobantes
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Calculer les centres des boîtes
        df_ball_positions['cx'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['cy'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['w'] = df_ball_positions['x2'] - df_ball_positions['x1']
        df_ball_positions['h'] = df_ball_positions['y2'] - df_ball_positions['y1']

        # Ajouter une dimension temporelle (index des frames)
        df_ball_positions['frame'] = np.arange(len(df_ball_positions))

        # Interpoler les positions manquantes avec CubicSpline
        interpolated_positions = {}
        for col in ['cx', 'cy', 'w', 'h']:
            valid_data = df_ball_positions[col].notna()
            frames_valid = df_ball_positions.loc[valid_data, 'frame']
            values_valid = df_ball_positions.loc[valid_data, col]

            if len(frames_valid) >= 3:  # CubicSpline nécessite au moins 3 points
                spline = CubicSpline(frames_valid, values_valid)
                interpolated_positions[col] = spline(df_ball_positions['frame'])
            else:  # Si pas assez de points pour une spline, utiliser une interpolation linéaire
                interpolated_positions[col] = df_ball_positions[col].interpolate(method='linear').bfill()

        # Reconstruire les boîtes englobantes après interpolation
        df_ball_positions['cx'] = interpolated_positions['cx']
        df_ball_positions['cy'] = interpolated_positions['cy']
        df_ball_positions['w'] = interpolated_positions['w']
        df_ball_positions['h'] = interpolated_positions['h']
        df_ball_positions['x1'] = df_ball_positions['cx'] - df_ball_positions['w'] / 2
        df_ball_positions['y1'] = df_ball_positions['cy'] - df_ball_positions['h'] / 2
        df_ball_positions['x2'] = df_ball_positions['cx'] + df_ball_positions['w'] / 2
        df_ball_positions['y2'] = df_ball_positions['cy'] + df_ball_positions['h'] / 2

        # Retourner le format original des positions
        ball_positions = [{1: {"bbox": row[['x1', 'y1', 'x2', 'y2']].tolist()}} for _, row in df_ball_positions.iterrows()]
        return ball_positions
    
    def interpolate_ball_positions_rnn(self, ball_positions):
        # Extraire les coordonnées des boîtes englobantes
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Calculer les centres et dimensions
        df['cx'] = (df['x1'] + df['x2']) / 2
        df['cy'] = (df['y1'] + df['y2']) / 2
        df['w'] = df['x2'] - df['x1']
        df['h'] = df['y2'] - df['y1']

        # Identifier les indices des données manquantes
        missing_indices = df[df.isnull().any(axis=1)].index

        # Normalisation des données
        scaler = MinMaxScaler()
        data = df[['cx', 'cy', 'w', 'h']].to_numpy()
        data_scaled = scaler.fit_transform(data)

        # Remplir les lacunes avec une interpolation pour entraîner un modèle
        data_filled = pd.DataFrame(data_scaled).interpolate(method='linear').bfill().to_numpy()

        # Préparer les séquences pour l'entraînement
        seq_length = 10  # Longueur des séquences
        sequences = []
        targets = []
        for i in range(len(data_filled) - seq_length):
            sequences.append(data_filled[i:i + seq_length])
            targets.append(data_filled[i + seq_length])

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Conversion en tenseurs PyTorch
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Initialiser et entraîner le modèle
        model = BallTrajectoryRNN()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs[:, -1, :], targets)
            loss.backward()
            optimizer.step()

        # Préparer les séquences pour les indices manquants
        missing_sequences = []
        for idx in missing_indices:
            if idx >= seq_length:
                seq = data_filled[idx - seq_length:idx]
                missing_sequences.append(seq)

        if missing_sequences:
            missing_sequences = torch.tensor(missing_sequences, dtype=torch.float32)

            # Prédire les valeurs manquantes
            missing_predicted = model(missing_sequences).detach().numpy()[:, -1, :]
            missing_predicted = scaler.inverse_transform(missing_predicted)

            # Remettre les prédictions dans le DataFrame original
            for idx, pred in zip(missing_indices, missing_predicted):
                df.at[idx, 'cx'] = pred[0]
                df.at[idx, 'cy'] = pred[1]
                df.at[idx, 'w'] = pred[2]
                df.at[idx, 'h'] = pred[3]

        # Reconstruction des boîtes englobantes
        df['x1'] = ((df['cx'] - df['w'] / 2).fillna(0)).astype(int)
        df['y1'] = ((df['cy'] - df['h'] / 2).fillna(0)).astype(int)
        df['x2'] = ((df['cx'] + df['w'] / 2).fillna(0)).astype(int)
        df['y2'] = ((df['cy'] + df['h'] / 2).fillna(0)).astype(int)

        # Retourner les positions mises à jour
        ball_positions = [{1: {"bbox": row[['x1', 'y1', 'x2', 'y2']].tolist()}} for _, row in df.iterrows()]
        return ball_positions

    def interpolate_ball_positions(self,ball_positions):
    
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position