from utils import read_video, save_video
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from tracker import BallTracker, FieldTracker, Tracker
import numpy as np



def analyse_video(video_path):
    # Load the YOLOv5 model
    video_name = video_path.split('/')[-1]
    video_frames,height, width = read_video(video_path)

    # Create a tracker object
    tracker = Tracker('weights/best.pt')
    # model line tracking football-field-detection-f07vi/15
    tracks = tracker.get_object_tracks(video_frames, read_from_strub=True, stub_path='stubs/'+video_name+'_track_stub.pkl')
    tracker.add_position_to_tracks(tracks)
    field_tracker = FieldTracker('weights/best_field-detector.pt')
    field_tracks = field_tracker.track_field_lines(video_frames,read_from_strub=True,stub_path='stubs/'+video_name+'_field_track_stub.pkl')
    # camera movement estimation
    #camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    #camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/'+video_name+'_camera_movement_stub.pkl')
    #camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    # Interpolate Ball Positions
    print(tracks["ball"])
    #tracks["ball"] = interpolate_ball_positions(tracks["ball"])
    ball_tracker = BallTracker()
    #ball_tracker.train_rnn(tracks["ball"], seq_length=20)  # Entraîner le modèle avec les données historiques
    #tracks["ball"] = ball_tracker.process_ball_positions(tracks["ball"]) 
    tracks["ball"] = ball_tracker.interpolate_ball_positions(tracks["ball"])
    tracks["ball"] = ball_tracker.interpolate_ball_positions_cubic_spline(tracks["ball"])
    #track["ball"] = ball_tracker.interpolate_ball_positions_cubic_spline(tracks["ball"])
    # Assign Player Teams
    # Initialize TeamAssigner
    team_assigner = TeamAssigner()

# Assign team colors by fitting the classifier on player crops from the first frame
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

# Assign teams to each track
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

    # Set team_colors in TeamAssigner
    team_assigner.team_colors = team_colors

    # Draw annotations

    #Assign Ball Aquistion
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
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


    #Draw output
    output_video_frames = field_tracker.annotate_field_lines(video_frames, field_tracks['field_lines'])
    #output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    

    #output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    # Save the video with the bounding boxes
    save_video(output_video_frames, 'output/video_output.mp4', height, width)

def get_average_color(image,intensity_factor=1.5):
    top_half_image = image[0:int(image.shape[0]/2), :]
    average_color = np.mean(top_half_image, axis=(0, 1))
    
    # Amplify the color intensit
    
    return average_color