from utils import read_video, save_video
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from tracker import BallTracker, FieldTracker, Tracker
import numpy as np
import pickle
import pandas as pd
import event_analysis
import cv2
import os


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
    # model line tracking football-field-detection-f07vi/15
    tracks = tracker.get_object_tracks(video_frames, read_from_strub=True, stub_path='stubs/'+video_name+'_track_stub.pkl')
    tracker.add_position_to_tracks(tracks)
    field_tracker = FieldTracker('weights/best_field-detector.pt')
    field_tracks = field_tracker.track_field_lines(video_frames,read_from_strub=True,stub_path='stubs/'+video_name+'_field_track_stub.pkl')
    print(field_tracks)
    # camera movement estimation
    # Interpolate Ball Positions
    #tracks["ball"] = interpolate_ball_positions(tracks["ball"])
    #camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    #camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    #camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/'+video_name+'_camera_movement_stub.pkl')
    ball_tracker = BallTracker()
    tracks["ball"] = ball_tracker.interpolate_ball_positions(tracks["ball"])
    tracks["ball"] = ball_tracker.interpolate_ball_positions_cubic_spline(tracks["ball"])
    # Assign Player Teams
    # Initialize TeamAssigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])


# Assign teams to each track
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        frame = video_frames[frame_num]
        for player_id, track in player_track.items():
            bbox = track['bbox']
            team = team_assigner.get_player_team(frame, bbox, player_id)
            tracks['players'][frame_num][player_id]['team'] = team


# Collect player colors grouped by team
    team_player_colors = {}
    for frame_num, player_track in enumerate(tracks['players']):
        frame = video_frames[frame_num]
        for player_id, track in player_track.items():
            team = track['team']
            bbox = track['bbox']
            image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            color = get_average_color(image,intensity_factor=0.5)
            if team not in team_player_colors:
                team_player_colors[team] = []
            team_player_colors[team].append(color)

# Compute average team colors
    team_colors = {}
    for team, colors in team_player_colors.items():
        team_colors[team] = np.mean(colors, axis=0)

# Assign team_colors to each track
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = track['team']
            tracks['players'][frame_num][player_id]['team_color'] = team_colors[team]
    team_assigner.team_colors = team_colors
    # Set team_colors in TeamAssigner


    # Draw annotations

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        if frame_num in tracks['ball'] and 1 in tracks['ball'][frame_num] and 'bbox' in tracks['ball'][frame_num][1]:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        else:
            continue 

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
    
       # Display and save paused frames
    print("Affichage des frames en pause...")
    display_paused_frames(video_path, pauses, video_frames, tracks, output_folder="output_frames")
    print("Affichage terminé.")


    #Draw output
    output_video_frames = field_tracker.annotate_field_lines(video_frames, field_tracks['field_lines'])
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, pauses)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    # Save the video with the bounding boxes
    save_video(output_video_frames, 'output/video_output.mp4', height, width)

def get_average_color(image,intensity_factor=1.5):
    top_half_image = image[0:int(image.shape[0]/2), :]
    average_color = np.mean(top_half_image, axis=(0, 1))
    
    # Amplify the color intensit
    
    return average_color

def display_paused_frames(video_path, pauses,video_frames, tracks,  output_folder="output_frames", show_frames=True):
    """
    Affiche et enregistre les frames où un arrêt de jeu est détecté.
    Désactive cv2.imshow() si aucun affichage graphique n'est disponible.
    """
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ Dossier de sortie créé : {output_folder}")

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erreur : Impossible d'ouvrir la vidéo à l'emplacement {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"📹 Vidéo chargée : {frame_count} frames, {fps} FPS")

    # Vérifier si l'affichage graphique est disponible (désactiver imshow si nécessaire)
    no_display = os.environ.get("DISPLAY") is None  # Vérifie si DISPLAY est défini (Linux/macOS)
    
    # Parcourir les pauses détectées
    for i, (start, end) in enumerate(pauses):
        print(f"🔍 Traitement de la pause {i} : frames {start} à {end}")
        for frame_num in range(start, min(end + 1, frame_count)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Erreur : Impossible de lire la frame {frame_num}")
                continue

            # Enregistrer la frame
            frame_path = os.path.join(output_folder, f"pause_{i}_frame_{frame_num}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"✅ Image enregistrée : {frame_path}")

            # Afficher la frame uniquement si un affichage est disponible
            if not no_display and show_frames:
                cv2.imshow(f"Pause {i} - Frame {frame_num}", frame)
                cv2.waitKey(300)

    cap.release()
    cv2.destroyAllWindows()



    # Event detection
    events_df = event_analysis.parse_xml_results('Annotations_AtomicEvents_Results.xml')
    ground_df = event_analysis.parse_ground_truth('Annotations_AtomicEvents_Results.xml')


 



    


