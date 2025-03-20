import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Function to detect and align face using MTCNN
def detect_and_crop_face(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        raise Exception("No face detected in the image")
    
    # Assume largest face is the target (based on bounding box area)
    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face['box']
    
    # Add some margin to the bounding box
    margin = int(max(w, h) * 0.2)
    x, y = max(0, x - margin), max(0, y - margin)
    w, h = w + 2 * margin, h + 2 * margin
    
    # Crop the face
    face_img = img_rgb[y:y+h, x:x+w]
    
    # Resize to 160x160 (FaceNet input size)
    face_img = cv2.resize(face_img, (160, 160))
    
    # Normalize pixel values to [-1, 1]
    face_img = (face_img / 255.0) * 2.0 - 1.0
    
    # Add batch dimension
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Function to get embeddings from FaceNet
def get_embedding(model, image_path):
    preprocessed_img = detect_and_crop_face(image_path)
    embedding = model.predict(preprocessed_img)
    return embedding[0]

# Load FaceNet model
print("Loading FaceNet model...")
facenet_model = load_model('FaceNet/facenet_keras_2024.h5')

# Dataset directory
dataset_dir = 'dataset'

# Prepare data
embeddings = []
labels = []

print("Processing images and generating embeddings...")
# Iterate through each person's folder
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_dir):
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            try:
                # Generate embedding
                embedding = get_embedding(facenet_model, image_path)
                embeddings.append(embedding)
                labels.append(person_name)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split dataset into training and testing (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, encoded_labels, test_size=0.2, random_state=42
)

# Train SVM classifier
print("Training SVM classifier...")
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)

# Test the model
print("Testing model...")
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"\nClassification Accuracy: {accuracy * 100:.2f}%")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of unique persons: {len(label_encoder.classes_)}")

# Optional: Print accuracy for each class
print("\nPer-class accuracy:")
for i, class_name in enumerate(label_encoder.classes_):
    class_mask = (y_test == i)
    if np.sum(class_mask) > 0:  # Only if there are samples in test set
        class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
        print(f"{class_name}: {class_acc * 100:.2f}%")
