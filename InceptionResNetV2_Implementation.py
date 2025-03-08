import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from mtcnn.mtcnn import MTCNN

# Create directory for saving face embeddings
if not os.path.exists('embeddings'):
    os.makedirs('embeddings')

# This approach uses a version of FaceNet that doesn't require loading a pre-trained .h5 file
# Instead, we'll use tf.keras.applications or a lightweight face embedding implementation

# Load InceptionResNetV2 as alternative to FaceNet
# This is not exactly FaceNet but can work for face embeddings
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore

# Create a model that outputs face embeddings
base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(160, 160, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
embedding_model = Model(inputs=base_model.input, outputs=x)

# Initialize MTCNN for face detection (more accurate than Haar cascades)
face_detector = MTCNN()

# Normalize images
l2_normalizer = Normalizer('l2')

def preprocess_face(face_img):
    """Preprocess face image for embedding model input"""
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    # Convert BGR to RGB (OpenCV uses BGR, but our model expects RGB)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Apply InceptionResNetV2 preprocessing
    face_img = preprocess_input(face_img)
    # Expand dimensions for model input
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def get_embedding(face_img):
    """Extract embedding vector from face image"""
    preprocessed = preprocess_face(face_img)
    embedding = embedding_model.predict(preprocessed)[0]
    # Normalize embedding vector
    embedding = l2_normalizer.transform(embedding.reshape(1, -1))[0]
    return embedding

def register_face(name):
    """Register a new face with name"""
    print(f"Registering face for {name}. Please look at the camera.")
    cap = cv2.VideoCapture(0)
    
    # Wait for 2 seconds before capturing to give user time to prepare
    start_time = time.time()
    while time.time() - start_time < 2:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam")
            return None
        
        cv2.putText(frame, f"Capturing in {int(3-(time.time()-start_time))}", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) == 27:  # ESC key to abort
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    # Capture and process the face
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    
    if not ret:
        print("Failed to capture image from webcam")
        return None
    
    # Detect face using MTCNN
    faces = face_detector.detect_faces(frame)
    
    if len(faces) == 0:
        print("No face detected")
        return None
    
    # Take the first detected face
    face_info = faces[0]
    x, y, w, h = face_info['box']
    face_img = frame[y:y+h, x:x+w]
    
    # Get and save embedding
    embedding = get_embedding(face_img)
    np.save(f"embeddings/{name}.npy", embedding)
    
    # Save thumbnail image
    cv2.imwrite(f"embeddings/{name}.jpg", face_img)
    
    print(f"Face registered for {name}")
    return embedding

def load_registered_faces():
    """Load all registered face embeddings"""
    registered_faces = {}
    
    if not os.path.exists('embeddings'):
        return registered_faces
    
    for file in os.listdir('embeddings'):
        if file.endswith('.npy'):
            name = os.path.splitext(file)[0]
            embedding = np.load(f"embeddings/{file}")
            registered_faces[name] = embedding
    
    return registered_faces

def recognize_face(embedding, registered_faces, threshold=0.6):
    """Compare face embedding to registered faces"""
    if not registered_faces:
        return "Unknown", 0
    
    max_similarity = -1
    best_match = "Unknown"
    
    for name, ref_embedding in registered_faces.items():
        similarity = cosine_similarity(embedding.reshape(1, -1), 
                                      ref_embedding.reshape(1, -1))[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = name
    
    if max_similarity < threshold:
        return "Unknown", max_similarity
        
    return best_match, max_similarity

def main():
    print("Webcam Facial Recognition System")
    print("================================")
    print("1. Register a new face")
    print("2. Start recognition")
    print("3. Exit")
    
    registered_faces = load_registered_faces()
    print(f"\nLoaded {len(registered_faces)} registered faces.")
    
    while True:
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            name = input("Enter name for the new face: ")
            embedding = register_face(name)
            if embedding is not None:
                registered_faces[name] = embedding
        
        elif choice == '2':
            # Start webcam recognition
            cap = cv2.VideoCapture(0)
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture image from webcam")
                        break
                        
                    # Make a copy for drawing
                    display_frame = frame.copy()
                    
                    # Detect faces using MTCNN
                    faces = face_detector.detect_faces(frame)
                    
                    for face_info in faces:
                        # Extract face coordinates
                        x, y, w, h = face_info['box']
                        
                        # Extract and process face
                        face_img = frame[y:y+h, x:x+w]
                        try:
                            embedding = get_embedding(face_img)
                            name, confidence = recognize_face(embedding, registered_faces)
                            
                            # Draw rectangle and put name
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(display_frame, f"{name} ({confidence:.2f})", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        except Exception as e:
                            print(f"Error processing face: {str(e)}")
                    
                    # Show the frame
                    cv2.imshow('Facial Recognition', display_frame)
                    
                    # Break on ESC key
                    if cv2.waitKey(1) == 27:
                        break
                        1
            finally:
                cap.release()
                cv2.destroyAllWindows()
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()