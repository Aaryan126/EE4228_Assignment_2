import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from mtcnn.mtcnn import MTCNN
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

####### Change as per your paths
dataset_path = 'dataset'
model_path = 'facenet_keras_2024.h5'
output_model_path = 'Finetuned_FaceNet/finetuned_facenet_mtcnn.keras' 

# Parameters
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20

# Initialize MTCNN detector
detector = MTCNN()

# Load the pre-trained FaceNet model
base_model = load_model(model_path)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Get number of classes
class_names = sorted(os.listdir(dataset_path))
num_classes = len(class_names)

# Helper function to extract face from bounding box
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # Ensure coordinates are within image bounds
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:  # Invalid box
        return None
    face = img[y1:y2, x1:x2]
    return face

# Load and preprocess data with MTCNN face detection
def load_data():
    X, y = [], []
    skipped_images = 0
    for idx, person in enumerate(class_names):
        person_path = os.path.join(dataset_path, person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            # Load image with OpenCV
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load {img_path}, skipping.")
                skipped_images += 1
                continue
            
            # Convert to RGB for MTCNN
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector.detect_faces(rgb_img)
            if len(faces) > 0:
                # Use the largest face (by area)
                main_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                face_img = get_face(img, main_face['box'])
                
                if face_img is None:
                    print(f"Invalid face box in {img_path}, skipping.")
                    skipped_images += 1
                    continue
                
                # Resize and preprocess
                face_img = cv2.resize(face_img, IMG_SIZE)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_array = face_img.astype('float32') / 255.0  # Normalize to [0,1]
                
                # Debug: Confirm detection
                print(f"Detected face in {img_path}: box={main_face['box']}")
                
                X.append(face_array)
                y.append(idx)
            else:
                print(f"No face detected in {img_path}")
                skipped_images += 1
    print(f"Total images skipped: {skipped_images}")
    return np.array(X), np.array(y)

# Load dataset
X, y = load_data()

# Check if data is loaded successfully
if len(X) == 0:
    raise ValueError("No valid face data loaded. Check dataset or MTCNN detection.")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Add classification head
x = base_model.output
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create a callback that saves the model's weights during training
checkpoint_callback = ModelCheckpoint(
    filepath=output_model_path,
    save_best_only=True,  # Save only the best model based on validation loss
    monitor="val_loss",
    mode="min",
    verbose=1
)

# Custom callback to compute precision, recall, and F1-score after each epoch
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(X_val), axis=1)  # Get predicted class labels
        y_true = y_val

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)

        print(f"\nEpoch {epoch+1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Create new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy')

# Train the model
history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val), 
                    callbacks=[checkpoint_callback, MetricsCallback()])

# Save the fine-tuned model
# model.save(output_model_path)

# print(f"Model fine-tuned and saved as {output_model_path}")
# print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
# print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Predict on validation set
y_pred_finetune = np.argmax(model.predict(X_val), axis=1)
y_true = y_val

# Calculate metrics for the pretrained model
precision_finetune = precision_score(y_true, y_pred_finetune, average='macro')
recall_finetune = recall_score(y_true, y_pred_finetune, average='macro')
f1_finetune = f1_score(y_true, y_pred_finetune, average='macro')
accuracy_finetune = accuracy_score(y_true, y_pred_finetune)

print(f"Finetuned Model - Precision: {precision_finetune:.4f}, Recall: {recall_finetune:.4f}, F1-score: {f1_finetune:.4f}, Accuracy: {accuracy_finetune:.4f}")
