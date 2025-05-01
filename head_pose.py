import numpy as np
import cv2


def solve_head_pose(face_landmarks, image_width, image_height):
    # 3D model points of key facial landmarks (in mm)
    face_model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float32)

    # Corresponding landmark indices from MediaPipe Face Mesh
    landmark_indices = [1, 152, 33, 263, 61, 291]

    # Convert MediaPipe normalized landmarks to pixel coordinates
    image_points = np.array([
        [face_landmarks.landmark[i].x * image_width, face_landmarks.landmark[i].y * image_height]
        for i in landmark_indices
    ], dtype=np.float32)

    # Camera matrix (intrinsic parameters)
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Solve PnP to get rotation and translation vectors
    success, rotation_vec, translation_vec = cv2.solvePnP(
        face_model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Head pose box (for visualization)
    head_pose_box_points = np.array([
        (0.0, 0.0, 1000.0),
        (100.0, 100.0, 1000.0),
        (-100.0, 100.0, 1000.0),
        (-100.0, -100.0, 1000.0),
        (100.0, -100.0, 1000.0),
        (0.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (100.0, 0.0, 0.0)
    ], dtype=np.float32)

    # Project head pose box points onto the image
    projected_points, _ = cv2.projectPoints(
        head_pose_box_points,
        rotation_vec,
        translation_vec,
        camera_matrix,
        dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)

    # Convert rotation vector to Euler angles (pitch, yaw, roll)
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()


    return projected_points, (pitch, yaw, roll)

