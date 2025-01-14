from ultralytics import YOLO
from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import numpy as np
import pickle
import pandas as pd

def detect_game_pauses(tracks, ball_key='ball', threshold_seconds=0.5, fps=30):
    """
    Détecte les arrêts de jeu si la balle est immobile pendant plus de `threshold_seconds` secondes.
    
    Args:
        tracks (dict): Données de tracking (comme celles dans le fichier .pkl).
        ball_key (str): Clé utilisée pour identifier les données de la balle.
        threshold_seconds (float): Durée minimale (en secondes) pour qu'un arrêt de jeu soit détecté.
        fps (int): Images par seconde de la vidéo.
    
    Returns:
        list: Liste des plages de frames où un arrêt de jeu a été détecté.
    """
    pauses = []
    pause_start = None
    consecutive_static_frames = 0
    threshold_frames = int(threshold_seconds * fps)

    # Vérifiez que tracks est un dictionnaire
    if not isinstance(tracks, dict):
        raise ValueError("tracks doit être un dictionnaire")

    # Récupérer les positions de la balle
    ball_positions = tracks.get(ball_key, [])
    if not ball_positions:
        print("Aucune donnée de position de la balle trouvée dans les tracks.")
        return pauses

    # Extraire les positions de la balle en tant que DataFrame
    ball_positions = [frame_data.get(1, {}).get('bbox', []) for frame_data in ball_positions]
    if not any(ball_positions):
        print("Les données de la balle sont vides.")
        return pauses

    # Créer un DataFrame avec les positions de la balle
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df_ball_positions['frame_num'] = range(len(ball_positions))  # Ajouter les numéros de frame
    print("Positions de la balle (DataFrame) :")
    print(df_ball_positions)

    # Vérifier les déplacements de la balle
    for i in range(1, len(df_ball_positions)):
        prev_center = np.array([(df_ball_positions.loc[i - 1, 'x1'] + df_ball_positions.loc[i - 1, 'x2']) / 2,
                                (df_ball_positions.loc[i - 1, 'y1'] + df_ball_positions.loc[i - 1, 'y2']) / 2])
        curr_center = np.array([(df_ball_positions.loc[i, 'x1'] + df_ball_positions.loc[i, 'x2']) / 2,
                                (df_ball_positions.loc[i, 'y1'] + df_ball_positions.loc[i, 'y2']) / 2])
        movement = np.linalg.norm(curr_center - prev_center)

        print(f"Frame {df_ball_positions.loc[i, 'frame_num']}: Movement = {movement}")

        # Vérifier si la balle est immobile (ou presque)
        if movement < 10:  # Réduire le seuil de mouvement
            consecutive_static_frames += 1
            if pause_start is None:
                pause_start = df_ball_positions.loc[i - 1, 'frame_num']
        else:
            # Si un mouvement est détecté, vérifier la durée de la pause
            if consecutive_static_frames >= threshold_frames:
                pause_end = df_ball_positions.loc[i - 1, 'frame_num']
                pauses.append((pause_start, pause_end))
            consecutive_static_frames = 0
            pause_start = None

    # Ajouter la dernière pause si elle n'a pas été enregistrée
    if consecutive_static_frames >= threshold_frames:
        pause_end = df_ball_positions.loc[len(df_ball_positions) - 1, 'frame_num']
        pauses.append((pause_start, pause_end))

    return pauses



def analyse_video(video_path):
    # Load the YOLOv5 model
    video_name = video_path.split('/')[-1]
    video_frames, height, width = read_video(video_path)

    # Create a tracker object
    tracker = Tracker('weights/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_strub=True, stub_path='stubs/' + video_name + '_track_stub.pkl')
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/' + video_name + '_camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracks["ball"] = tracker.interpolate_ball_positions_rnn(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Detect game pauses
    print("Détection des arrêts de jeu...")
    pauses = detect_game_pauses(tracks, ball_key="ball", threshold_seconds=1, fps=30)
    print("Arrêts de jeu détectés :", pauses)
    for start, end in pauses:
        duration = (end - start) / 30  # Convertir les frames en secondes
        print(f"Arrêt de jeu entre les frames {start} et {end} (durée : {duration:.2f} secondes)")

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Save the video with the bounding boxes
    save_video(output_video_frames, 'output/video_output.mp4', height, width)
