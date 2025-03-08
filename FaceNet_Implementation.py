import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Create directory for saving face embeddings
if not os.path.exists('embeddings'):
    os.makedirs('embeddings')

# Path to the FaceNet model - you'll need to set this to your actual path
# You'll need to download this separately
MODEL_PATH = 'facenet_keras_2024.h5'  # Update this path

# Load the FaceNet model
# If you run into compatibility issues, you'll need to use the right TF version
# or convert the model as discussed
facenet_model = load_model(MODEL_PATH)
print("FaceNet model loaded successfully.")

# Initialize MTCNN detector
detector = MTCNN()

# Initialize the L2 normalizer
l2_normalizer = Normalizer('l2')

def get_face(img, box):
    """Extract face from image based on bounding box"""
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # Extract the face
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def preprocess_face(face_img):
    """Preprocess face for FaceNet input"""
    # Resize to the expected size
    face_img = cv2.resize(face_img, (160, 160))
    
    # Convert to RGB (if it's BGR from OpenCV)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert data type to float32
    face_img = face_img.astype(np.float32)
    
    # Normalize pixel values
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    
    # Expand dimensions to create a batch of size 1
    face_img = np.expand_dims(face_img, axis=0)
    
    return face_img

def get_embedding(face_img):
    """Generate embedding vector from face image"""
    # Preprocess the face image
    face_img = preprocess_face(face_img)
    
    # Generate embedding
    embedding = facenet_model.predict(face_img)[0]
    
    # Normalize embedding
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
    
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    # Detect faces using MTCNN
    results = detector.detect_faces(frame)
    
    if not results:
        print("No face detected. Please try again.")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    # Get the largest face (assuming it's the main person)
    largest_face = sorted(results, key=lambda x: x['box'][2] * x['box'][3], reverse=True)[0]
    face_img, (x1, y1), (x2, y2) = get_face(frame, largest_face['box'])
    
    # Draw rectangle around the face
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show the captured face
    cv2.imshow('Captured Face', frame)
    cv2.waitKey(1000)  # Display for 1 second
    
    # Generate and save embedding
    try:
        embedding = get_embedding(face_img)
        np.save(f"embeddings/{name}.npy", embedding)
        
        # Save face image
        cv2.imwrite(f"embeddings/{name}.jpg", face_img)
        
        print(f"Face successfully registered for {name}")
        
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    cap.release()
    cv2.destroyAllWindows()
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

def recognize_face(embedding, registered_faces, threshold=0.5):
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
    print("FaceNet Facial Recognition with MTCNN")
    print("=====================================")
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
            fps_start_time = time.time()
            fps_counter = 0
            fps = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture image from webcam")
                        break
                    
                    # Calculate FPS
                    fps_counter += 1
                    if time.time() - fps_start_time >= 1.0:
                        fps = fps_counter
                        fps_counter = 0
                        fps_start_time = time.time()
                    
                    # Make a copy for drawing
                    display_frame = frame.copy()
                    
                    # Detect faces
                    try:
                        faces = detector.detect_faces(frame)
                        
                        for face_info in faces:
                            # Extract face
                            face_img, (x1, y1), (x2, y2) = get_face(frame, face_info['box'])
                            
                            # Get embedding and recognize
                            embedding = get_embedding(face_img)
                            name, confidence = recognize_face(embedding, registered_faces)
                            
                            # Determine color based on match (green for known, red for unknown)
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            
                            # Draw rectangle and name
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, f"{name} ({confidence:.2f})", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    except Exception as e:
                        print(f"Error in face detection/recognition: {str(e)}")
                    
                    # Display FPS
                    cv2.putText(display_frame, f"FPS: {fps}", (20, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show frame
                    cv2.imshow('Facial Recognition', display_frame)
                    
                    # Exit on ESC
                    if cv2.waitKey(1) == 27:
                        break
            
            finally:
                cap.release()
                cv2.destroyAllWindows()
        
        elif choice == '3':
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()