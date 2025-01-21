from sklearn.cluster import KMeans

from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
from sklearn.cluster import KMeans
import sentencepiece
from transformers import AutoProcessor, SiglipVisionModel
import umap

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch

class TeamClassifier:
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in batches:
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

class TeamAssigner:
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.team_classifier = TeamClassifier(device=device, batch_size=batch_size)
        self.player_team_dict = {}

    def collect_player_crops(self, frame, player_detections):
        player_crops = []
        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            top_half_image = image[0:int(image.shape[0]/2), :]
            player_crops.append(top_half_image)
        return player_crops

    def assign_team_color(self, frame, player_detections):
        player_crops = self.collect_player_crops(frame, player_detections)
        self.team_classifier.fit(player_crops)

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        image = frame[int(player_bbox[1]):int(player_bbox[3]), int(player_bbox[0]):int(player_bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2), :]
        player_crop = [top_half_image]
        team_label = self.team_classifier.predict(player_crop)[0]
        team_id = team_label + 1  # Assuming team labels start from 1

        # Override for specific player_id if needed
        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id