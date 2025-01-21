from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS

import cv2
import numpy as np

import sys
from ultralytics import YOLO
from sklearn.cluster import KMeans
import analysis

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_video():
    uploaded_file = request.files['video']

    if uploaded_file.filename != '':
        # Sauvegardez la vidéo dans un répertoire sur votre serveur
        uploaded_file.save('uploads/footballvideo.mp4')
        video_path = 'uploads/footballvideo.mp4'
        analysis.analyse_video(video_path)

        # Vous pouvez maintenant traiter la vidéo ou renvoyer une réponse si nécessaire
        return {'message': 'Video uploaded successfully'}, 200
    else:
        return {'error': 'No file selected'}, 400

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    directory = "output"
    print(filename)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/view/<filename>', methods=['GET'])
def view_file(filename):
    directory = "output"
    print(filename)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run()
