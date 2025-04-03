import cv2
from models import *
import re

# metrics for face matching
metric = "euclidean"
threshold = 150
# metric = "cosine"
# threshold = 0.000015
print(f"Metric funtion is {metric} with threshold: {threshold}\n")

print("Connecting to camera...")
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
print("Camera connected...\n")

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

frame_count = 0
process_every_n_frames = 1

while True:
	# Grab a single frame of video
	_, frame = video_capture.read()

	frame_count += 1  # Increment frame counter
	if frame_count % process_every_n_frames == 0:  # Process every n frames
		
		# Find all the faces and face encodings in the current frame of video
		face_locations = dlib_get_face_locations(frame)
		face_encodings = dlib_cvt2_encodings(frame)
		# face_locations = MTCNN_get_face_locations(frame)
		# face_encodings = MTCNN_cvt2_encodings(frame)

		face_names = []
		for face_encoding in face_encodings:
			# Use the known face with the smallest distance to the new face
			face_distances = face_distance(known_face_encodings, face_encoding, metric)
			best_match_index = np.argmin(face_distances)
			print(face_distances[best_match_index])
			if face_distances[best_match_index] <= threshold:
				name = known_face_labels[best_match_index]

			else:
				name = "Unknown"

			match = re.search(r"^(\w+)_", name)
			name = match.group(1) if match else name
			face_names.append(name)

		if face_encodings.any():
			# Display the results
			for (x_min, y_min, x_max, y_max), name in zip(face_locations, face_names):
				cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
				cv2.putText(frame, name, (x_min, y_min - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
