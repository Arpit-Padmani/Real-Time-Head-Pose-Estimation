"""
Real-Time Head Pose Estimation
--------------------------------
Webcam-based head pose (yaw / pitch / roll) estimation using MediaPipe
FaceLandmarker + OpenCV solvePnP. Features a live 3D point-cloud 
reconstruction panel and directional tracking.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ------------------------------------------------------------------ #
# CONFIGURATION & SETUP
# ------------------------------------------------------------------ #
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

CANVAS_SIZE = 500
LANDMARK_IDS = [33, 263, 1, 61, 291, 199]

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading FaceLandmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

download_model()

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO,
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# ------------------------------------------------------------------ #
# 3D PANEL RENDERING
# ------------------------------------------------------------------ #
def render_3d_face_panel(face_landmarks, img_w, img_h, rmat):
    """Renders the 3D face mesh and X, Y, Z axes properly."""
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    origin = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)

    nose_lm = face_landmarks[1]
    
    # Static 3/4 camera view to make it look 3D (30 degrees around Y axis)
    view_angle = np.radians(30)
    view_rot = np.array([
        [np.cos(view_angle), 0, np.sin(view_angle)],
        [0, 1, 0],
        [-np.sin(view_angle), 0, np.cos(view_angle)],
    ])

    # 1. Draw the face mesh dots (No rmat applied here to avoid double-rotation!)
    for lm in face_landmarks:
        # Center points around the nose
        px = (lm.x - nose_lm.x) * img_w
        py = (lm.y - nose_lm.y) * img_h
        pz = (lm.z - nose_lm.z) * 3000
        
        # Apply only the static camera view
        rotated = view_rot @ np.array([px, py, pz])

        sx = int(origin[0] + rotated[0])
        sy = int(origin[1] + rotated[1])
        if 0 <= sx < CANVAS_SIZE and 0 <= sy < CANVAS_SIZE:
            cv2.circle(canvas, (sx, sy), 1, (0, 255, 0), -1)

    # 2. Draw the X, Y, Z axes using rmat (Head Rotation) + view_rot (Camera View)
    axis_len = 100
    base_axes = np.array([
        [axis_len, 0, 0], # X (Red)
        [0, axis_len, 0], # Y (Green)
        [0, 0, axis_len]  # Z (Blue)
    ])

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labels = ["X", "Y", "Z"]

    for i in range(3):
        # Rotate axis by head rotation, then by static camera view
        head_axis = rmat @ base_axes[i]
        final_axis = view_rot @ head_axis
        
        pt = (int(origin[0] + final_axis[0]), int(origin[1] + final_axis[1]))
        
        cv2.line(canvas, origin, pt, colors[i], 3)
        cv2.putText(canvas, labels[i], pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    cv2.putText(canvas, "3D Face Reconstruction (3/4 View)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return canvas

# ------------------------------------------------------------------ #
# MAIN LOOP
# ------------------------------------------------------------------ #
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    frame_timestamp_ms = int(time.time() * 1000)
    results = face_mesh.detect_for_video(mp_image, frame_timestamp_ms)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = image.shape

    face_panel = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            face_2d = []
            face_3d = []
            nose_2d = None

            # EXACT LOGIC from your old code
            for idx, lm in enumerate(face_landmarks):
                if idx in LANDMARK_IDS:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    if idx == 1:
                        nose_2d = (x, y)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]], dtype=np.float64)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, *_ = cv2.RQDecomp3x3(rmat)

            # Old angle multipliers
            pitch_x = angles[0] * 360
            yaw_y = angles[1] * 360
            roll_z = angles[2] * 360

            # Fixed Direction Logic (Swapped Up/Down)
            if yaw_y < -10:
                text = "Looking Left"
            elif yaw_y > 10:
                text = "Looking Right"
            elif pitch_x < -10:
                text = "Looking Up"
            elif pitch_x > 10:
                text = "Looking Down"
            else:
                text = "Forward"
            
            # --- Draw Old Blue Line logic on Webcam ---
            if nose_2d:
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + yaw_y * 10), int(nose_2d[1] - pitch_x * 10))
                cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Draw green face mesh dots
            for lm in face_landmarks:
                x_px, y_px = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(image, (x_px, y_px), 1, (0, 255, 0), -1)

            # Draw text
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(image, f"pitch(x): {pitch_x:.1f}", (img_w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"yaw(y):   {yaw_y:.1f}", (img_w - 250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"roll(z):  {roll_z:.1f}", (img_w - 250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Render 3D Screen
            face_panel = render_3d_face_panel(face_landmarks, img_w, img_h, rmat)

    fps = 1 / (time.time() - start + 1e-6)
    cv2.putText(image, f'FPS: {int(fps)}', (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Head Pose Tracker (Webcam)", image)
    cv2.imshow("3D Face Reconstruction", face_panel)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()