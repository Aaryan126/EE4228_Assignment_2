import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime

# Configuration paths
#MODEL_PATH = 'Finetuned_FaceNet/finetuned_facenet.h5'
def scaling(x, scale=0.17):
    return x * scale

# Load the model and pass custom objects
#finetuned_model = load_model("Finetuned_FaceNet/facenet_finetuned_tf.keras", custom_objects={"scaling": scaling})
MODEL_PATH = 'Finetuned_FaceNet/finetuned_facenet.keras'
EMBEDDINGS_PATH = 'Finetuned_FaceNet/saved_embeddings.pkl' #Mathematical Transformation of the images
REGISTERED_IMAGES_PATH = 'Finetuned_FaceNet/registered_images.txt'

# Load FaceNet model
facenet_model = load_model(MODEL_PATH, custom_objects={"scaling": scaling})
print("FaceNet model loaded successfully.")

# Initialize components
detector = MTCNN()
l2_normalizer = Normalizer('l2')

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img - face_img.mean()) / face_img.std()
    return np.expand_dims(face_img, axis=0)

def get_embedding(face_img):
    embedding = facenet_model.predict(preprocess_face(face_img))[0]
    return l2_normalizer.transform(embedding.reshape(1, -1))[0]

def register_faces_from_folder(dataset_path="dataset"):
    registered_faces = {}
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            embeddings = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)
                    faces = detector.detect_faces(img)
                    if faces:
                        main_face = max(faces, key=lambda x: x['box'][2]*x['box'][3])
                        face_img, _, _ = get_face(img, main_face['box'])
                        try:
                            embeddings.append(get_embedding(face_img))
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")
            if embeddings:
                registered_faces[person_name] = np.array(embeddings)
                print(f"Registered {len(embeddings)} embeddings for {person_name}")
    return registered_faces

def recognize_face(embedding, registered_faces, threshold=0.5):
    if not registered_faces:
        return "Unknown", 0
    similarities = {
        name: np.max(cosine_similarity([embedding], embeddings))
        for name, embeddings in registered_faces.items()
    }
    best_match = max(similarities, key=similarities.get)
    return (best_match, similarities[best_match]) if similarities[best_match] > threshold else ("Unknown", 0)

# Embeddings
def save_embeddings(embeddings):
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")

def load_embeddings():
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def check_dataset_modified(dataset_path):
    try:
        with open(REGISTERED_IMAGES_PATH, 'r') as f:
            saved_files = {line.split('|')[0]: float(line.split('|')[1]) for line in f.read().splitlines()}
    except FileNotFoundError:
        return True

    current_files = {}
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                current_files[path] = os.path.getmtime(path)
    
    return saved_files != current_files

def update_registered_images(dataset_path):
    registered = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                registered.append(f"{path}|{os.path.getmtime(path)}")
    with open(REGISTERED_IMAGES_PATH, 'w') as f:
        f.write('\n'.join(registered))

def main():
    print("Face Recognition System")
    print("1. Register Faces (Train)")
    print("2. Start Recognition")
    print("3. Save Current Model")
    print("4. Exit")
    
    registered_faces = load_embeddings() or {}
    if registered_faces:
        print("\nPre-loaded embeddings:")
        for name, data in registered_faces.items():
            print(f"→ {name} ({len(data)} embeddings)")

    while True:
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == '1':
            dataset_path = input("Dataset path [dataset]: ").strip() or "dataset"
            if not check_dataset_modified(dataset_path):
                print("Using existing embeddings (dataset unchanged)")
                registered_faces = load_embeddings()
                continue
                
            registered_faces = register_faces_from_folder(dataset_path)
            if registered_faces:
                save_embeddings(registered_faces)
                update_registered_images(dataset_path)
        
        elif choice == '2':
            if not registered_faces:
                print("No registered faces! Train first.")
                continue
            
            cap = cv2.VideoCapture(0)
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(rgb_frame)
                    
                    for face in faces:
                        face_img, (x1, y1), (x2, y2) = get_face(frame, face['box'])
                        embedding = get_embedding(face_img)
                        name, confidence = recognize_face(embedding, registered_faces)
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) == 27:
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
        
        elif choice == '3':
            if registered_faces:
                save_embeddings(registered_faces)
                print("Model saved successfully!")
            else:
                print("No embeddings to save!")
        
        elif choice == '4':
            print("Exiting system...")
            break
        
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
