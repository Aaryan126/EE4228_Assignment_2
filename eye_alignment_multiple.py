import cv2
import dlib
import numpy as np
from imutils import face_utils

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./checkpoints/shape_predictor_68_face_landmarks.dat")

def align_faces(image):
    """Detects and aligns multiple faces in the image using dlib."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No face detected!")
        return [], []  # Return empty lists if no faces are found

    aligned_faces = []
    bounding_boxes = []

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Get eye positions
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Compute center points
        left_center = left_eye.mean(axis=0).astype("int")
        right_center = right_eye.mean(axis=0).astype("int")

        # Compute the rotation angle
        dY = right_center[1] - left_center[1]
        dX = right_center[0] - left_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Compute center point for rotation
        center = ((left_center[0] + right_center[0]) / 2.0,
                  (left_center[1] + right_center[1]) / 2.0)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Get dimensions of the image
        (h, w) = image.shape[:2]

        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # Get bounding box of the rotated face
        x_min = max(face.left(), 0)
        y_min = max(face.top(), 0)
        x_max = min(face.right(), w - 1)
        y_max = min(face.bottom(), h - 1)

        # Crop the rotated face
        cropped_face = rotated[y_min:y_max, x_min:x_max]

        # Ensure valid crop
        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
            print("Invalid face crop detected, skipping.")
            continue  # Skip invalid detections

        # Resize cropped face to a fixed size (e.g., 200x200)
        cropped_face = cv2.resize(cropped_face, (200, 200))

        aligned_faces.append(cropped_face)
        bounding_boxes.append((x_min, y_min, x_max, y_max))

    return aligned_faces, bounding_boxes  # Return all faces and their bounding boxes
