import cv2
import face_recognition
import os
import pickle
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = "models/yolov11n.pt"
EMBEDDINGS_PATH = "embeddings/face_encodings.pkl"

model = YOLO(MODEL_PATH)

if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings, known_face_names = [], []

def process_video(video_path, report_path):
    cap = cv2.VideoCapture(video_path)
    report_data = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                face_img = frame[y1:y2, x1:x2]
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)

                name = "Unknown"
                if encodings:
                    encoding = encodings[0]
                    matches = face_recognition.compare_faces(known_face_encodings, encoding)
                    if True in matches:
                        match_index = matches.index(True)
                        name = known_face_names[match_index]

                report_data.append({
                    "frame": frame_count,
                    "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                    "name": name
                })

    cap.release()
    df = pd.DataFrame(report_data)
    df.to_csv(report_path, index=False)
