import dlib

def numpy_to_dlib_landmark(landmark_array, image_shape):
    """
    Convert a NumPy array of landmarks (from MTCNN or another detector)
    into a dlib.full_object_detection object.
    
    :param landmark_array: NumPy array of shape (68, 2) (x, y coordinates).
    :param image_shape: Tuple (height, width) of the image.
    :return: dlib.full_object_detection object.
    """
    if landmark_array is None or len(landmark_array) == 0:
        raise ValueError("Empty or invalid landmark array")

    # Create a dlib rectangle covering the entire face
    rect = dlib.rectangle(left=0, top=0, right=image_shape[1], bottom=image_shape[0])

    # Convert NumPy landmarks into dlib format
    dlib_landmarks = dlib.full_object_detection(
        rect, dlib.points([dlib.point(int(x), int(y)) for x, y in landmark_array])
    )

    return dlib_landmarks