from ultralytics import YOLO
from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import numpy as np

def analyse_video(video_path):
    # Load the YOLOv5 model
    video_name = video_path.split('/')[-1]
    video_frames,height, width = read_video(video_path)

    # Create a tracker object
    tracker = Tracker('weights/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_strub=True, stub_path='stubs/'+video_name+'_track_stub.pkl')
    tracker.add_position_to_tracks(tracks)
    # camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/'+video_name+'_camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracks["ball"] = tracker.interpolate_ball_positions_rnn(tracks["ball"]) 

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Assign Ball Aquistion
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


    #Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    # Save the video with the bounding boxes
    save_video(output_video_frames, 'output/video_output.mp4', height, width)