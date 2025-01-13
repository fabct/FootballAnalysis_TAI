import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames, height, width

def save_video(ouput_video_frames,output_video_path,height,width):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc,
                                   30.0,
                                   (width, height))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
