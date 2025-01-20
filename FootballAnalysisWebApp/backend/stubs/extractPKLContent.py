import pickle
with open('footballvideo.mp4_track_stub.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)