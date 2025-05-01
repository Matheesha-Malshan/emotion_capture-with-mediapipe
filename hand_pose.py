import numpy as np
import hand_pose as hp

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def is_hand_near_face(hand_landmarks, face_landmarks, img_w, img_h):
    hand_tip = hand_landmarks[8]  # index fingertip
    nose = face_landmarks[1]      # nose tip

    hx, hy = hand_tip.x * img_w, hand_tip.y * img_h
    nx, ny = nose.x * img_w, nose.y * img_h

    distan=distance((hx, hy), (nx, ny)) 
    return distan
