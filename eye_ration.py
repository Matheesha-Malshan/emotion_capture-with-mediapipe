import numpy as np

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, image_width, image_height):
    coords = [(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in eye_indices]
    A = euclidean(coords[1], coords[5])
    B = euclidean(coords[2], coords[4])
    C = euclidean(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
