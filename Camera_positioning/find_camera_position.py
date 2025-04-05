import cv2
import numpy as np
import json
import os

camera_index = 2
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1820)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Optionally, force 4K resolution if supported:
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Load camera calibration data.
with open('camera_calibration_data.json', 'r') as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)

# Load marker database (with known 3D marker corners).
marker_db_file = 'aruco_marker_positions.json'
if os.path.exists(marker_db_file):
    with open(marker_db_file, 'r') as f:
        marker_db = json.load(f)
else:
    print("Marker database file not found. Exiting.")
    exit()

# Setup ArUco detector.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Create a full-screen window.
window_name = "Auto Marker Pose with Trident"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Global variable to hold the computed pose.
computed_pose = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for marker detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    # Draw marker outlines and IDs.
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

    # Automatically compute pose if any markers are detected.
    if ids is not None and len(ids) > 0:
        all_obj_points = []
        all_img_points = []
        for i, marker in enumerate(ids):
            marker_id = str(marker[0])
            if marker_id in marker_db:
                data = marker_db[marker_id]["corner_points"]
                corners_world = np.array([
                    [data["top_left"]["x"],     data["top_left"]["y"],     data["top_left"]["z"]],
                    [data["top_right"]["x"],    data["top_right"]["y"],    data["top_right"]["z"]],
                    [data["bottom_right"]["x"], data["bottom_right"]["y"], data["bottom_right"]["z"]],
                    [data["bottom_left"]["x"],  data["bottom_left"]["y"],  data["bottom_left"]["z"]]
                ], dtype=np.float32)
                all_obj_points.append(corners_world)
                all_img_points.append(corners[i][0])
        if len(all_obj_points) > 0:
            all_obj_points = np.vstack(all_obj_points)
            all_img_points = np.vstack(all_img_points)
            ret_pnp, rvec, tvec = cv2.solvePnP(all_obj_points, all_img_points,
                                               camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
            if ret_pnp:
                computed_pose = (rvec, tvec)
                # Draw the coordinate axes (trident) at the world origin.
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)

                # Compute the camera position.
                R, _ = cv2.Rodrigues(rvec)
                cam_pos = -R.T.dot(tvec)
                # Compute yaw (heading) by projecting the optical axis [0,0,1].
                forward_vector = R.T.dot(np.array([[0], [0], [1]], dtype=np.float32))
                yaw = np.degrees(np.arctan2(forward_vector[1,0], forward_vector[0,0]))
                if yaw < 0:
                    yaw += 360

                # Overlay camera position and yaw at the bottom of the frame.
                pos_text = f"Pos: X={cam_pos[0,0]:.1f}, Y={cam_pos[1,0]:.1f}, Z={cam_pos[2,0]:.1f}"
                yaw_text = f"Yaw: {yaw:.1f} deg"
                cv2.putText(frame, pos_text, (20, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, yaw_text, (20, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc key to exit.
        break

cap.release()
cv2.destroyAllWindows()
