from ultralytics import YOLO
import supervision as sv
from pykalman import KalmanFilter
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
    
    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dist = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            # Draw players
            for track_id, player in player_dist.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_ellipse(frame, player['bbox'], (0,255,0), track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,255))

            for _, ball in ball_dict.items():
                frame = self.draw_rectangle(frame, ball['bbox'], (255,0,0),'ball')

            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)

            output_video_frames.append(frame)
        
        return output_video_frames
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of times each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        total_frames = team_1_num_frames + team_2_num_frames

        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = 0
            team_2 = 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

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

        return frame
    
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

    def apply_kalman_filter(self, data):
        # Assurez-vous que 'data' est un tableau NumPy
        ball_positions = [x.get(1, {}).get("bbox", []) for x in data]
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Créez un Kalman Filter avec l'état initial approprié
        kf = KalmanFilter(initial_state_mean=np.zeros(df.shape[1]),
                      n_dim_obs=df.shape[1])
    
        # Appliquez le filtre Kalman
        smoothed_state_means, _ = kf.smooth(df)
    
        df["x1"], df["y1"], df["x2"], df["y2"] = smoothed_state_means[:, 0], smoothed_state_means[:, 1], smoothed_state_means[:, 2], smoothed_state_means[:, 3]

        # Retourner les positions mises à jour
        return [{1: {"bbox": row[["x1", "y1", "x2", "y2"]].tolist()}} for _, row in df.iterrows()]