import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tqdm import tqdm

# Disable OneDNN optimizations to avoid potential compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1️⃣ Load the Pretrained FaceNet Model with Progress Bar
print("🔄 Loading FaceNet Model...")
with tqdm(total=1, desc="Loading FaceNet", unit="step") as pbar:
    facenet_model = load_model("FaceNet/facenet_keras_2024.h5")
    pbar.update(1)
print("✅ FaceNet Model Loaded!")

# 2️⃣ Prepare Your Dataset (Organized as "dataset/person_name/*.jpg")
dataset_path = "dataset"

print("🔄 Loading dataset from:", dataset_path)
train_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=(160, 160),  # FaceNet expects 160x160 images
    batch_size=32
)

# Display class names (for verification)
class_names = train_dataset.class_names
print(f"✅ Dataset loaded! Found {len(class_names)} classes.")

# 3️⃣ Modify the Model for Fine-tuning
x = facenet_model.output  # FaceNet already provides 128-dimensional embeddings
output_layer = Dense(len(class_names), activation='softmax')(x)  # Classification head

# Create the new fine-tuned model
finetuned_model = Model(inputs=facenet_model.input, outputs=output_layer)

# 4️⃣ Compile the Model
finetuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])

# 5️⃣ Train the Model with Progress Bar
epochs = 5  # Adjust based on need
print(f"🔄 Training model for {epochs} epochs...")

for epoch in range(epochs):
    print(f"🟢 Epoch {epoch+1}/{epochs}")
    with tqdm(total=len(train_dataset), desc=f"Training Progress (Epoch {epoch+1})", unit="batch") as pbar:
        history = finetuned_model.fit(train_dataset, epochs=1)
        pbar.update(len(train_dataset))

print("✅ Training Complete!")

# 6️⃣ Save the Entire Model with Progress Bar
save_path = "Finetuned_FaceNet/facenet_finetuned_tf.keras"
#save_path = "test_finetuned_model.keras"
print(f"🔄 Saving model to {save_path}...")

with tqdm(total=1, desc="Saving Model", unit="step") as pbar:
    finetuned_model.save(save_path)
    pbar.update(1)

print("✅ Model saved successfully in TensorFlow SavedModel format!")
