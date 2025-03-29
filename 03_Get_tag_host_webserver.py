from flask import Flask, jsonify
import cv2
import numpy as np
import json
import math

app = Flask(__name__)

def rotationMatrixToEulerAngles(R):
    """
    Converts a rotation matrix to Euler angles (roll, pitch, yaw).
    The yaw is the rotation around the Z-axis.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0

    return np.array([roll, pitch, yaw])

# ------------- Load Camera Calibration and Pose -------------
with open('camera_calibration_data.json', 'r') as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)

with open('camera_pose.json', 'r') as f:
    pose_data = json.load(f)
rvec_cam = np.array(pose_data["rotation_vector"], dtype=np.float32)
tvec_cam = np.array(pose_data["translation_vector"], dtype=np.float32)

R_cam, _ = cv2.Rodrigues(rvec_cam)
R_cam_inv = R_cam.T

# ------------- Marker Model -------------
marker_size = 232  # Adjust as needed
marker_obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# ------------- ArUco Detector Setup -------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

@app.route('/robot1')
def robot1():
    # Initialize the webcam
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    # Convert frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None:
        return jsonify({"error": "No markers detected"}), 404

    # Look for marker with id 1
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == 1:
            img_pts = corners[i][0].astype(np.float32)
            ret_pnp, rvec_marker, tvec_marker = cv2.solvePnP(
                marker_obj_points, img_pts, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ret_pnp:
                return jsonify({"error": "solvePnP failed for marker id 1"}), 500

            # Compute global marker position
            t_marker_world = R_cam_inv @ (tvec_marker - tvec_cam)
            # Compute yaw angle in global frame
            R_marker, _ = cv2.Rodrigues(rvec_marker)
            R_marker_global = R_cam_inv @ R_marker
            euler_angles = rotationMatrixToEulerAngles(R_marker_global)
            yaw_deg = np.degrees(euler_angles[2])

            result = {
                "id": int(marker_id),
                "global_position": t_marker_world.flatten().tolist(),
                "yaw": yaw_deg
            }
            return jsonify(result)

    # If marker with id 1 was not found in the frame
    return jsonify({"error": "Marker with id 1 not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
