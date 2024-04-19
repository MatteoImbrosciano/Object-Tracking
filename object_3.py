import cv2
import torch
from tracker import ObjectTracker  # Assumi che il codice della classe ObjectTracker sia salvato in un file tracker.py
import pandas as pd
import os

# Carica il modello pre-addestrato YOLOv5
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    exit(1)

# Crea un'istanza di ObjectTracker specifica per le automobili
tracker = ObjectTracker(max_distance=50, max_missed_frames=10)

# Percorso al video delle automobili
video_path = "C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\prog\\bdd100k\\00c29c52-f9524f1e.mp4"

# Imposta il percorso per il video di output con gli oggetti tracciati
tracked_video_path = "tracked_video_3.mp4"
video_writer = None

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Errore nell'apertura del video")
    exit(1)

class_id_for_car = 2  # ID classe per "car" nel modello YOLOv5, controlla per essere sicuro
detected_ids = set()
tracked_results = {}  # Dizionario per memorizzare i risultati del tracking

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Estrai i risultati

        frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Ottieni l'indice del frame corrente
        tracked_results[int(frame_index)] = []

        rects = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], int(det[5])
            if conf > 0.3 and cls == class_id_for_car:
                rects.append([x1, y1, x2 - x1, y2 - y1])
                tracked_results[int(frame_index)].append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Aggiorna il tracker con i nuovi rilevamenti
        objects = tracker.update(rects)

        for obj in objects:
            obj_id = obj[-1]
            if obj_id not in detected_ids:
                detected_ids.add(obj_id)

        cv2.putText(frame, f'Cars in this frame: {len(objects)}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Scrivi il frame elaborato nel video di output
        if video_writer is None:
            frame_height, frame_width = frame.shape[:2]
            video_writer = cv2.VideoWriter(tracked_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
        video_writer.write(frame)

        cv2.imshow('Processed Frame', frame)

        if cv2.waitKey(30) == 27:  # Esc per uscire
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# Salva i risultati in un DataFrame e quindi in un CSV
df_tracked = pd.DataFrame([(k, *v) for k, vals in tracked_results.items() for v in vals], columns=['frameIndex', 'box2d.x1', 'box2d.y1', 'box2d.x2', 'box2d.y2'])
df_tracked.to_csv('tracked_results_3.csv', index=False)
