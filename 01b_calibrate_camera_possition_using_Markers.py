import cv2
import numpy as np
import json
import os
from datetime import datetime

camera_index = 2

# ----- Load Camera Calibration Data -----
with open('camera_calibration_data.json', 'r') as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)

# ----- Load Marker Database -----
marker_db_file = 'aruco_marker_positions.json'
if os.path.exists(marker_db_file):
    with open(marker_db_file, 'r') as f:
        marker_db = json.load(f)
else:
    print("Marker database file not found. Exiting.")
    exit()

# ----- Setup ArUco Detector -----
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# ----- Setup Video Capture and Window -----

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
window_name = "Marker Mode Pose Estimation"
cv2.namedWindow(window_name)

# Global variable for computed pose (rvec, tvec)
computed_pose = None

# Instructions to overlay on the stream.
instructions = (
    "Marker Mode\n"
    " - Detected markers (in DB) will be drawn\n"
    " - Press 'p' to compute pose\n"
    " - Press 's' to save pose\n"
    " - Press Esc to exit"
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        break

    # Convert frame to grayscale for marker detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    # If markers are detected, process each one.
    if ids is not None:
        for i, marker in enumerate(ids):
            marker_id = str(marker[0])
            if marker_id in marker_db:
                pts = np.int32(corners[i][0])
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                for pt in pts:
                    cv2.circle(frame, tuple(pt), 4, (0, 255, 255), -1)
                cxy = pts.mean(axis=0).astype(int)
                cv2.putText(frame, marker_id, (cxy[0]-10, cxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If a pose has been computed, overlay the global trident and camera info.
    if computed_pose is not None:
        rvec, tvec = computed_pose
        # Draw coordinate axes (trident) at the world origin.
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)
        # Compute camera position in world coordinates.
        R, _ = cv2.Rodrigues(rvec)
        cam_pos = -R.T.dot(tvec)
        cam_pos_flat = cam_pos.flatten()
        # Compute the camera's forward (optical) direction in world coordinates.
        forward_vector = R.T.dot(np.array([[0], [0], [1]], dtype=np.float32))
        heading_angle = np.degrees(np.arctan2(forward_vector[1, 0], forward_vector[0, 0]))
        if heading_angle < 0:
            heading_angle += 360
        pos_text = f"Cam Pos: X={cam_pos_flat[0]:.1f}, Y={cam_pos_flat[1]:.1f}, Z={cam_pos_flat[2]:.1f}"
        heading_text = f"Heading: {heading_angle:.1f} deg"
        cv2.putText(frame, pos_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, heading_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Overlay instructions.
    y0 = 20
    for i, line in enumerate(instructions.split("\n")):
        cv2.putText(frame, line, (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit on Esc key.
    if key == 27:
        break

    # Pose estimation on 'p' key.
    elif key == ord('p'):
        if ids is None or len(ids) == 0:
            print("No markers detected for pose estimation.")
            continue

        all_obj_points = []
        all_img_points = []
        # For each detected marker in the frame that exists in our DB,
        # get its known 3D corner points and corresponding 2D image points.
        for i, marker in enumerate(ids):
            marker_id = str(marker[0])
            if marker_id in marker_db:
                data = marker_db[marker_id]["corner_points"]
                corners_world = np.array([
                    [data["top_left"]["x"], data["top_left"]["y"], data["top_left"]["z"]],
                    [data["top_right"]["x"], data["top_right"]["y"], data["top_right"]["z"]],
                    [data["bottom_right"]["x"], data["bottom_right"]["y"], data["bottom_right"]["z"]],
                    [data["bottom_left"]["x"], data["bottom_left"]["y"], data["bottom_left"]["z"]]
                ], dtype=np.float32)
                all_obj_points.append(corners_world)
                all_img_points.append(corners[i][0])
        if len(all_obj_points) == 0:
            print("No known markers found for pose estimation.")
            continue

        all_obj_points = np.vstack(all_obj_points)
        all_img_points = np.vstack(all_img_points)

        ret, rvec, tvec = cv2.solvePnP(all_obj_points, all_img_points,
                                       camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if ret:
            print("Pose estimated:")
            print("Rotation Vector:", rvec.flatten())
            print("Translation Vector:", tvec.flatten())
            # Store computed pose for continuous overlay.
            computed_pose = (rvec, tvec)
        else:
            print("solvePnP failed.")

    # Save computed pose on 's' key.
    elif key == ord('s'):
        if computed_pose is not None:
            new_pose = {
                "rotation_vector": computed_pose[0].tolist(),
                "translation_vector": computed_pose[1].tolist()
            }
            with open("camera_pose.json", "w") as f:
                json.dump(new_pose, f, indent=4)
            print("Pose saved to 'camera_pose.json':", new_pose)
        else:
            print("No pose computed to save.")

cap.release()
cv2.destroyAllWindows()
