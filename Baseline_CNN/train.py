import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, utils, callbacks, Input
import tensorflow as tf

# Configuration
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "baseline_cnn_embeddings.h5"
INPUT_SHAPE = (100, 100, 3)
BATCH_SIZE = 32
EPOCHS = 20

# Initialize components
detector = MTCNN()

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    return img[y1:y2, x1:x2]

def load_dataset():
    faces = []
    labels = []
    label_names = []

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)
        if os.path.isdir(person_path):
            label_names.append(person_name)
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Face detection
                    detected_faces = detector.detect_faces(rgb_img)
                    if detected_faces:
                        main_face = max(detected_faces, 
                                      key=lambda x: x['box'][2]*x['box'][3])
                        face_img = get_face(img, main_face['box'])
                        
                        # Preprocessing
                        face_img = cv2.resize(face_img, (100, 100))
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_img = (face_img - face_img.mean()) / face_img.std()
                        
                        faces.append(face_img)
                        labels.append(person_name)
    
    return np.array(faces), np.array(labels), label_names

def create_model(num_classes):
    inputs = Input(shape=INPUT_SHAPE)  # Explicit input layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    embeddings = layers.Dense(128, name='embeddings')(x)  # Embedding layer
    outputs = layers.Dense(num_classes, activation='softmax')(embeddings)

    model = models.Model(inputs=inputs, outputs=outputs)  # Functional model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Load and prepare dataset
    X, y, class_names = load_dataset()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Create model
    model = create_model(len(class_names))
    
    # Add callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )
    
    # Save embedding model
    embedding_model = models.Model(
        inputs=model.input,
        outputs=model.get_layer('embeddings').output
    )
    embedding_model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()