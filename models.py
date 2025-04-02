import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import cv2
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler

# using dlib for face detection and alignment
from eye_alignment_multiple import align_faces	

def save_data():
	np.savez(file_name, known_face_encodings, known_face_labels)

def load_image_file(file):
	# im = Image.open(file)
	# im = im.convert("RGB")
	image = cv2.imread(file)
	return image

def face_distance(encodings, encoding, metric):
    if len(encodings) == 0:
        return np.empty(0)

    if metric == "euclidean":
        return np.linalg.norm(encodings - encoding, axis=1)
    else:
        a1 = np.sum(np.multiply(encodings, encoding), axis=1)
        b1 = np.sum(np.multiply(encodings, encodings), axis=1)
        c1 = np.sum(np.multiply([encoding], [encoding]), axis=1)
        return (1 - (a1 / (b1**.5 * c1**.5)))

def MTCNN_face_detector(frame):
	face_detector = MTCNN()

	face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	faces = face_detector.detect_faces(face_image)

	return faces

def MTCNN_cvt2_encodings(frame):
	faces = MTCNN_face_detector(frame)
	standardScaler = StandardScaler()
	if not faces:
		return np.array([])
	encodings = []  # Store multiple face encodings
	for face in faces:
		x, y, width, height = face["box"]
		x1, y1 = max(0, x), max(0, y)
		x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)

		# Crop the face region
		cropped_face = frame[y1:y2, x1:x2]

		# Resize, normalize, and prepare for the model
		resized_face = cv2.resize(cropped_face, (224, 224))
		# resized_face = resized_face.reshape(1, -1) 
		# resized_face = standardScaler.fit_transform(resized_face)
		resized_face = resized_face.reshape(1, 224, 224, 3)
		# Get face encoding using ResNet50
		face_encoding = np.squeeze(resnet50_features(resized_face))

		encodings.append(face_encoding)
	return np.array(encodings)

def MTCNN_get_face_locations(frame):
	faces = MTCNN_face_detector(frame)
	if not faces:
		return []

	face_locations = []
	for face in faces:
		x, y, w, h = face["box"]
		x_min = x
		y_min = y
		x_max = x + w
		y_max = y + h
		face_locations.append((x_min, y_min, x_max, y_max))

	return face_locations

def dlib_cvt2_encodings(frame):
	aligned_faces, bounding_boxes = align_faces(frame)  # from your custom eye_alignment_multiple
	encodings = []  # Store multiple face encodings

	for face in aligned_faces:
		resized_face = cv2.resize(face, (224, 224))
		resized_face = resized_face.reshape(1, 224, 224, 3)
		face_encoding = np.squeeze(resnet50_features(resized_face))
		encodings.append(face_encoding)
	return np.array(encodings)

def dlib_get_face_locations(frame):
	aligned_faces, bounding_boxes = align_faces(frame)  # from your custom eye_alignment_multiple
	if not align_faces:
		return []
	return bounding_boxes

def add_image(image_path):
	global known_face_labels, known_face_encodings, changed

	root, _ = os.path.splitext(image_path)
	label = os.path.split(root)[-1]

	if not np.isin(label, known_face_labels):
		print(f"Adding {label} ...")
		image = load_image_file(image_path)
		image_encodings = dlib_cvt2_encodings(image)		# change depending which face detector you want to use
		# image_encodings = MTCNN_cvt2_encodings(image)


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


# define base path for the project
BASE = r"C:/Users/jy158/Desktop/NTU/Notes/Y4S2/EE4228 Intelligent System Design/Assignment/real-time-one-shot-face-recognition/"

# location to store embeddings
file_name = BASE + "encodings/database.npz"
changed = False
print(f"Database file: {file_name}\n")

# load resnet50 model for feature extraction
checkpoint_path = BASE + "checkpoints/resnet50_face_recognition.h5"
if os.path.exists(checkpoint_path):
	base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
	resnet50_features = Model(inputs=base_model.input, outputs=base_model.output)
	print(f"Loading checkpoint from `{checkpoint_path}`...")
	resnet50_features.load_weights(checkpoint_path, by_name=True)
	print("Model checkpoint loaded.")
else:
	print("No checkpoint found. Initializing a new model...")
	resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

print("Embedding extraction model loaded...\n")

try:
	known_face_encodings, known_face_labels = np.load(file_name).values()
except IOError:
	known_face_encodings, known_face_labels = np.array([]), np.array([], "str")
	changed = True

## add images for new members
for dir, _, files in os.walk("../Member Photos"):
	for file in files:
		add_image(os.path.join(dir, file))

# saving updated encoding on close
if changed:
	save_data()

print(f"Total Faces in database: {len(known_face_labels)}\n")
# print(f"Encoding shape in database: {known_face_encodings.shape}")
