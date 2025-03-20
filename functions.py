import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import cv2
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

BASE = r"C:/Users/jy158/Desktop/NTU/Notes/Y4S2/EE4228 Intelligent System Design/Assignment/real-time-one-shot-face-recognition/"
checkpoint_path = BASE + "checkpoints/resnet50_face_recognition.h5"

file_name = BASE + "encodings/database.npz"
changed = False
print(f"Database file: {file_name}")

# metric = "cosine"
# threshold = 0.000015
metric = "euclidean"
threshold= 80
print(f"metric funtion is {metric} and threshold {threshold}")

face_detector = MTCNN()
print("Face detector model loded...")

# Check if a checkpoint exists and load the model
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from `{checkpoint_path}`...")
    resnet50_features = load_model(checkpoint_path)
    resnet50_features.trainable = False
    print("Model checkpoint loaded.")
else:
    print("No checkpoint found. Initializing a new model...")
    resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling="avg")

print("Embedding extraction model loaded...")

EXPORT_FILE = "data.csv"
export_data = []

try:
    known_face_encodings, known_face_labels = np.load(file_name).values()
except IOError:
	known_face_encodings, known_face_labels = np.array([]), np.array([], "str")
	changed = True


def save_data():
	np.savez(file_name, known_face_encodings, known_face_labels)


def save_export():
	from pandas import DataFrame, concat

	df = DataFrame(export_data, columns=["Name", "Time"])
	df = concat([
		df.drop_duplicates(subset=["Name"], keep='first'),
		df.drop_duplicates(subset=["Name"], keep='last'),
	])
	df.to_csv(EXPORT_FILE, index=False)
	print(f"Data saved to - `{EXPORT_FILE}`")


def load_image_file(file):
	# im = Image.open(file)
	# im = im.convert("RGB")
	image = cv2.imread(file)
	return image

def face_distance(encodings, encoding):
    if len(encodings) == 0:
        return np.empty(0)

    if metric == "euclidean":
        return np.linalg.norm(encodings - encoding, axis=1)
    else:
        a1 = np.sum(np.multiply(encodings, encoding), axis=1)
        b1 = np.sum(np.multiply(encodings, encodings), axis=1)
        c1 = np.sum(np.multiply([encoding], [encoding]), axis=1)
        return (1 - (a1 / (b1**.5 * c1**.5)))

def get_face_encodings(face_image, known_face_locations=None, num_jitters=1):
	face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
	faces = face_detector.detect_faces(face_image)
	standardScaler = StandardScaler()
	if not faces:
		return np.array([])
	encodings = []  # Store multiple face encodings
	for face in faces:
		x, y, width, height = face["box"]
		x1, y1 = max(0, x), max(0, y)
		x2, y2 = min(face_image.shape[1], x + width), min(face_image.shape[0], y + height)

		# Crop the face region
		cropped_face = face_image[y1:y2, x1:x2]

		# Resize, normalize, and prepare for the model
		resized_face = cv2.resize(cropped_face, (224, 224))
		# resized_face = resized_face.reshape(1, -1) 
		# resized_face = standardScaler.fit_transform(resized_face)
		resized_face = resized_face.reshape(1, 224, 224, 3)
		# Get face encoding using ResNet50
		face_encoding = np.squeeze(resnet50_features(resized_face))

		encodings.append(face_encoding)
	return np.array(encodings)

def get_face_locations(img):
    # Convert to RGB (face detector might expect this format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_detector.detect_faces(img)
    if not faces:
        return []

    face_locations = []
    for face in faces:
        x, y, w, h = face["box"]
        top = y
        right = x + w
        bottom = y + h
        left = x
        face_locations.append((top, right, bottom, left))
    
    return face_locations

def add_image(image_path):
	global known_face_labels, known_face_encodings, changed

	root, _ = os.path.splitext(image_path)
	label = os.path.split(root)[-1]

	if not np.isin(label, known_face_labels):
		print(f"Adding {label} ...")
		image = load_image_file(image_path)
		image_encodings = get_face_encodings(image)

		if not image_encodings.any():
			print(f"No face found in `{label}`, so not added.")
			return
		
		if image_encodings.shape[0] > 1:
			print(f"Multiple faces found in `{label}`, so not added.")
			return

		if known_face_labels.size == 0:
			known_face_encodings = np.array([image_encodings[0]])
			known_face_labels = np.array([label])
		else:
			known_face_encodings = np.vstack([known_face_encodings, np.expand_dims(image_encodings[0], axis=0)])
			known_face_labels = np.append(known_face_labels, label)
		print(f"Added {label}")
		
		changed = True
	# else:
	# 	print(f"Image `{label}` already exist with same name")

def remove_image(label):
	global known_face_labels, known_face_encodings, changed

	known_face_encodings = known_face_encodings[known_face_labels != label]
	known_face_labels = known_face_labels[known_face_labels != label]
	changed = True

## add images
for dir, _, files in os.walk("../Member Photos"):
	for file in files:
		add_image(os.path.join(dir, file))

# saving updated encoding on close
if changed:
	save_data()

print(f"Total Faces in database: {len(known_face_labels)}")
print(f"Encoding shape in database: {known_face_encodings.shape}")
