from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import torch.nn as nn
import torch
import sys
sys.path.append('../')
from ball_trajectory import BallTrajectoryRNN

class BallTracker:
    def __init__(self):
        self.model = BallTrajectoryRNN()
        self.scaler = MinMaxScaler()

    def train_rnn(self, tracks_ball, seq_length=10, epochs=500, lr=0.001):
        ball_positions = [x.get(1, {}).get("bbox", [np.nan, np.nan, np.nan, np.nan]) for x in tracks_ball]
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Étape 2 : Calcul des centres et dimensions
        df["cx"] = (df["x1"] + df["x2"]) / 2
        df["cy"] = (df["y1"] + df["y2"]) / 2
        df["w"] = df["x2"] - df["x1"]
        df["h"] = df["y2"] - df["y1"]

        # Étape 3 : Interpolation pour remplir les lacunes
        df = df.interpolate(method="linear").bfill()

        # Étape 4 : Normalisation des données
        scaler = MinMaxScaler()
        data = df[["cx", "cy", "w", "h"]].to_numpy()
        data_scaled = scaler.fit_transform(data)

        # Étape 5 : Préparation des séquences pour le RNN
        sequences = []
        targets = []
        for i in range(len(data_scaled) - seq_length):
            sequences.append(data_scaled[i : i + seq_length])
            targets.append(data_scaled[i + seq_length])

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Conversion en tenseurs PyTorch
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Étape 6 : Initialisation et entraînement du modèle
        model = BallTrajectoryRNN()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):  # Nombre d'époques ajustable
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs[:, -1, :], targets)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        print(f"Training complete. Final Loss: {loss.item()}")

    def interpolate_rnn(self, df, seq_length=10):
        # Identifier les indices des données manquantes
        missing_indices = df[df.isnull().any(axis=1)].index

        # Normalisation des données
        scaler = MinMaxScaler()
        data = df[['cx', 'cy', 'w', 'h']].to_numpy()
        data_scaled = scaler.fit_transform(data)

        # Remplir les lacunes avec une interpolation linéaire
        data_filled = pd.DataFrame(data_scaled).interpolate(method='linear').bfill().to_numpy()

        # Préparer les séquences pour la prédiction (sans apprentissage)
        sequences = []
        for i in range(len(data_filled) - seq_length):
            sequences.append(data_filled[i:i + seq_length])

        sequences = np.array(sequences)

        # Remplacer les valeurs manquantes par des valeurs interpolées
        for idx in missing_indices:
            if idx >= seq_length:
                seq = data_filled[idx - seq_length:idx]
                missing_pred = seq[-1]  # On prend la dernière valeur de la séquence comme prédiction

                # Remettre les prédictions dans le DataFrame original
                df.at[idx, 'cx'] = missing_pred[0]
                df.at[idx, 'cy'] = missing_pred[1]
                df.at[idx, 'w'] = missing_pred[2]
                df.at[idx, 'h'] = missing_pred[3]

        # Reconstruction des boîtes englobantes
        return df

    def interpolate_ball_positions_cubic_spline(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    
        # Calculate center and dimensions
        df['cx'] = (df['x1'] + df['x2']) / 2
        df['cy'] = (df['y1'] + df['y2']) / 2
        df['w'] = df['x2'] - df['x1']
        df['h'] = df['y2'] - df['y1']
    
        # Interpolate missing values using cubic splines
        for col in ['cx', 'cy', 'w', 'h']:
            df[col] = df[col].interpolate(method='cubic')
        
        # Reconstruct bounding boxes
        df['x1'] = ((df['cx'] - df['w'] / 2).fillna(0)).astype(int)
        df['y1'] = ((df['cy'] - df['h'] / 2).fillna(0)).astype(int)
        df['x2'] = ((df['cx'] + df['w'] / 2).fillna(0)).astype(int)
        df['y2'] = ((df['cy'] + df['h'] / 2).fillna(0)).astype(int)
    
        # Return the interpolated positions
        return [{1: {"bbox": row[['x1', 'y1', 'x2', 'y2']].tolist()}} for _, row in df.iterrows()]

    def apply_kalman_filter(self, data):
        # Ensure 'data' is a NumPy array
        data = np.array(data)
    
        # Define the Kalman filter
        kf = KalmanFilter(n_dim_obs=4, n_dim_state=4)
    
        # Initialize the filter with the first measurement
        kf = kf.em(data, n_iter=10)
    
        # Smooth the entire sequence
        smoothed_data, _ = kf.smooth(data)
    
        # Return the smoothed positions
        return smoothed_data

    def process_ball_positions(self, ball_positions):
        interpolated_positions = self.interpolate_ball_positions(ball_positions)

        # Vérification et ajout des valeurs par défaut si nécessaire
        ball_positions_array = []
        for pos in interpolated_positions:
            if isinstance(pos, dict) and 1 in pos and 'bbox' in pos[1]:
                ball_positions_array.append(pos[1]['bbox'])
            else:
                # Ajouter une entrée par défaut vide
                ball_positions_array.append([0, 0, 0, 0])  # Par exemple, une bbox nulle
    
        ball_positions_array = np.array(ball_positions_array)

        # Appliquer le filtre de Kalman
        smoothed_positions = self.apply_kalman_filter(ball_positions_array)

        # Reconstruire les positions avec les valeurs lissées
        smoothed_ball_positions = [
            {1: {"bbox": pos.tolist()}} if pos is not None else {}
            for pos in smoothed_positions
        ]

        return smoothed_ball_positions

    def interpolate_ball_positions(self, ball_tracks, threshold=20):
        """
        Interpolate missing ball positions in the ball_tracks list.

        Parameters:
        - ball_tracks: List containing frame-wise ball data.
        - threshold: Maximum number of consecutive frames to interpolate.

        Returns:
        - The updated ball_tracks list with interpolated positions.
        """
        detected = []
        for i, frame_data in enumerate(ball_tracks):
            if frame_data and 1 in frame_data and 'position' in frame_data[1] and 'bbox' in frame_data[1]:
                detected.append((i, frame_data[1]['position'], frame_data[1]['bbox']))

        for i in range(len(ball_tracks)):
            if not ball_tracks[i]:
                # Trouver les frames précédentes et suivantes détectées
                prev_idx, next_idx = -1, -1
                for j in range(i - 1, -1, -1):
                    if ball_tracks[j] and 1 in ball_tracks[j]:
                        prev_idx = j
                        break
                for j in range(i + 1, len(ball_tracks)):
                    if ball_tracks[j] and 1 in ball_tracks[j]:
                        next_idx = j
                        break

                if prev_idx != -1 and next_idx != -1:
                    frames_to_interpolate = next_idx - prev_idx - 1
                    if frames_to_interpolate <= threshold:
                        pos_prev = ball_tracks[prev_idx][1]['position']
                        pos_next = ball_tracks[next_idx][1]['position']
                        bbox_prev = ball_tracks[prev_idx][1]['bbox']
                        bbox_next = ball_tracks[next_idx][1]['bbox']
                        fraction = (i - prev_idx) / (next_idx - prev_idx)

                        # Interpolation de la position
                        interpolated_x = pos_prev[0] + (pos_next[0] - pos_prev[0]) * fraction
                        interpolated_y = pos_prev[1] + (pos_next[1] - pos_prev[1]) * fraction

                        # Interpolation de la bbox
                        interpolated_x1 = bbox_prev[0] + (bbox_next[0] - bbox_prev[0]) * fraction
                        interpolated_y1 = bbox_prev[1] + (bbox_next[1] - bbox_prev[1]) * fraction
                        interpolated_x2 = bbox_prev[2] + (bbox_next[2] - bbox_prev[2]) * fraction
                        interpolated_y2 = bbox_prev[3] + (bbox_next[3] - bbox_prev[3]) * fraction

                        # Vérification de la cohérence avec le mouvement précédent
                        if prev_idx > 0 and ball_tracks[prev_idx - 1] and 1 in ball_tracks[prev_idx - 1]:
                            pos_prev_prev = ball_tracks[prev_idx - 1][1]['position']
                            movement_prev = ((pos_prev[0] - pos_prev_prev[0]) ** 2 +
                                             (pos_prev[1] - pos_prev_prev[1]) ** 2) ** 0.5
                            threshold_movement = movement_prev * 2  # Ajuster le multiplicateur si nécessaire
                            movement_interpolated = ((interpolated_x - pos_prev[0]) ** 2 +
                                                     (interpolated_y - pos_prev[1]) ** 2) ** 0.5
                            if movement_interpolated <= threshold_movement:
                                ball_tracks[i] = {
                                    1: {
                                        'position': (interpolated_x, interpolated_y),
                                        'bbox': [interpolated_x1, interpolated_y1, interpolated_x2, interpolated_y2]
                                    }
                                }
                        else:
                            ball_tracks[i] = {
                                1: {
                                    'position': (interpolated_x, interpolated_y),
                                    'bbox': [interpolated_x1, interpolated_y1, interpolated_x2, interpolated_y2]
                                }
                            }
        return ball_tracks