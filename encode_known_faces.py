import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_PATH = "embeddings/face_encodings.pkl"

known_face_encodings = []
known_face_names = []

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    for filename in os.listdir(person_path):
        img_path = os.path.join(person_path, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(person_name)

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Encodings saved to", EMBEDDINGS_PATH)
