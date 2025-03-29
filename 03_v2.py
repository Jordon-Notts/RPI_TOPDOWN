import threading
import cv2
import numpy as np
import json
import math
from flask import Flask, jsonify

app = Flask(__name__)

# Global variable to hold the latest detection result and a lock for thread safety
detection_result = None
result_lock = threading.Lock()

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

# ------------- Define Marker Model & Global Axes (Trident) -------------
marker_size = 232  # adjust as needed

# Marker model: centered at (0,0,0)
marker_obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

axis_length = marker_size
axes_points_world = np.array([
    [0, 0, 0],
    [axis_length, 0, 0],
    [0, axis_length, 0],
    [0, 0, axis_length]  # Points upward (Z up)
], dtype=np.float32)

# ------------- Initialize Webcam & ArUco Detector -------------
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

def detection_loop():
    global detection_result
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # Process the detection if markers are found
        if ids is not None:
            # Optionally draw the detected markers for debugging
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # Process each detected marker
            for i, marker_id in enumerate(ids.flatten()):
                img_pts = corners[i][0].astype(np.float32)
                ret_pnp, rvec_marker, tvec_marker = cv2.solvePnP(
                    marker_obj_points, img_pts,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ret_pnp:
                    # Compute global marker position
                    t_marker_world = R_cam_inv @ (tvec_marker - tvec_cam)
                    # (Optional) Compute marker center for drawing
                    center = np.mean(img_pts, axis=0)
                    
                    # (Optional) Draw a circle at the marker center
                    cv2.circle(frame, tuple(np.int32(center)), 5, (0, 0, 0), -1)
                    
                    # (Optional) Draw an enclosing circle on the marker plane
                    num_points = 50
                    r = marker_size / 2
                    theta = np.linspace(0, 2 * np.pi, num_points)
                    circle_3d = np.stack((r * np.cos(theta), r * np.sin(theta), np.zeros(num_points)), axis=1).astype(np.float32)
                    circle_imgpts, _ = cv2.projectPoints(circle_3d, rvec_marker, tvec_marker, camera_matrix, dist_coeffs)
                    circle_imgpts = np.int32(circle_imgpts).reshape(-1, 2)
                    cv2.polylines(frame, [circle_imgpts], isClosed=True, color=(0,0,255), thickness=2)
                    
                    # (Optional) Draw an arrowed line for the top of the marker
                    top_point_marker = np.array([[0, marker_size/2, 0]], dtype=np.float32)
                    top_imgpts, _ = cv2.projectPoints(top_point_marker, rvec_marker, tvec_marker, camera_matrix, dist_coeffs)
                    top_imgpts = np.int32(top_imgpts).reshape(-1, 2)
                    cv2.arrowedLine(frame, tuple(np.int32(center)), tuple(top_imgpts[0]), (255, 0, 255), 2)
                    
                    # Compute yaw (rotation about global Z-axis)
                    R_marker, _ = cv2.Rodrigues(rvec_marker)
                    R_marker_global = R_cam_inv @ R_marker
                    euler_angles = rotationMatrixToEulerAngles(R_marker_global)
                    yaw_deg = np.degrees(euler_angles[2])
                    
                    # If marker id 1 is detected, update the global detection result
                    if marker_id == 1:
                        with result_lock:
                            detection_result = {
                                "id": int(marker_id),
                                "global_position": t_marker_world.flatten().tolist(),
                                "yaw": yaw_deg
                            }
                        print(f"Marker ID {marker_id} global pos: {t_marker_world.flatten()} Yaw: {yaw_deg:.1f} deg")
                else:
                    print(f"solvePnP failed for marker {marker_id}")
        else:
            # If no markers are detected, you might choose to clear or update the detection result
            with result_lock:
                detection_result = None

        # (Optional) Draw the Global Axes (Trident)
        imgpts_axes, _ = cv2.projectPoints(axes_points_world, rvec_cam, tvec_cam, camera_matrix, dist_coeffs)
        imgpts_axes = np.int32(imgpts_axes).reshape(-1, 2)
        cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[1]), (0, 0, 255), 2)
        cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[2]), (0, 255, 0), 2)
        cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[3]), (255, 0, 0), 2)
        cv2.circle(frame, tuple(imgpts_axes[0]), 3, (0, 0, 0), -1)
        
        # (Optional) Display the frame for debugging. If running headless, you can comment these out.
        cv2.imshow("Global 3D Markers & Trident", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit detection loop
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/robot')
def robot_data():
    with result_lock:
        if detection_result is not None:
            return jsonify(detection_result)
        else:
            return jsonify({"error": "No marker detected"}), 404

if __name__ == '__main__':
    # Start the background thread for continuous detection
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    # Run the Flask server (adjust host and port as needed)
    app.run(debug=True, host='0.0.0.0', port=5000)
