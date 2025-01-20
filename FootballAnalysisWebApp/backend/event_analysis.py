import xml.etree.ElementTree as ET
import pandas as pd
import cv2

def parse_xml_results(xml_path):
    """
    Parse le fichier XML pour extraire les événements et leurs performances.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    events_data = []
    
    for event in root.find('overall'):
        event_name = event.tag
        if event_name != 'windowSize':  # Éviter le champ non pertinent
            event_dict = {
                'event': event_name,
                'total': int(event.find('total').text),
                'TP': int(event.find('TP').text),
                'FP': int(event.find('FP').text),
                'FN': int(event.find('FN').text),
                'precision': float(event.find('precision').text) if event.find('precision') is not None else None,
                'recall': float(event.find('recall').text) if event.find('recall') is not None else None
            }
            events_data.append(event_dict)
    
    df = pd.DataFrame(events_data)
    return df

def parse_ground_truth(xml_path):
    """
    Extrait les événements du fichier XML pour vérifier leur correspondance avec le tracking.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    ground_events = []
    
    for event in root.find('ground'):
        event_dict = {
            'name': event.find('name').text,
            'track_id': int(event.find('track').text),
            'match': event.find('match').text == 'True'
        }
        ground_events.append(event_dict)
    
    df = pd.DataFrame(ground_events)
    return df


def display_paused_frames(video_path, pauses, output_folder="output_frames"):
    """
    Affiche et enregistre les frames où un arrêt de jeu est détecté.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    for i, (start, end) in enumerate(pauses):
        for frame_num in range(start, min(end + 1, frame_count)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(output_folder, f"pause_{i}_frame_{frame_num}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"✅ Image enregistrée : {frame_path}")

                cv2.imshow(f"Pause {i} - Frame {frame_num}", frame)
                cv2.waitKey(300)
    
    cap.release()
    cv2.destroyAllWindows()

# Charger les données XML
events_df = parse_xml_results('Annotations_AtomicEvents_Results.xml')
ground_df = parse_ground_truth('Annotations_AtomicEvents_Results.xml')

# Affichage des résultats

print("Évaluation des événements :")
print(events_df)

print("\nAnnotations des événements :")
print(ground_df)
