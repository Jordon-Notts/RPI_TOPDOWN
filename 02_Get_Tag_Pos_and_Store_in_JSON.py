import cv2
import numpy as np
import json
import math

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

# ------------- Step 2: Load Camera Calibration and Pose -------------
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

# ------------- Step 3: Define Marker Model & Global Axes (Trident) -------------
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
    [0, 0, axis_length]  # Now points upward (Z up)
], dtype=np.float32)

# ------------- Step 4: Initialize Webcam & ArUco Detector -------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

print("Starting marker detection. Press 'Esc' to exit.")

# ------------- Step 5: Main Loop -------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    # Dictionary to store marker positions and orientations
    markers_data = {}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            img_pts = corners[i][0].astype(np.float32)
            ret_pnp, rvec_marker, tvec_marker = cv2.solvePnP(marker_obj_points, img_pts,
                                                             camera_matrix, dist_coeffs,
                                                             flags=cv2.SOLVEPNP_ITERATIVE)
            if ret_pnp:
                # Compute global marker position
                t_marker_world = R_cam_inv @ (tvec_marker - tvec_cam)
                text = f"ID {marker_id}: [{t_marker_world.flatten()[0]:.1f}, {t_marker_world.flatten()[1]:.1f}, {t_marker_world.flatten()[2]:.1f}]"
                
                # Compute marker center from image corners
                center = np.mean(img_pts, axis=0)
                
                # Draw circle at marker center (filled black circle)
                cv2.circle(frame, tuple(np.int32(center)), 5, (0, 0, 0), -1)
                
                # Draw an enclosing circle on the marker plane:
                num_points = 50
                r = marker_size / 2
                theta = np.linspace(0, 2 * np.pi, num_points)
                circle_3d = np.stack((r * np.cos(theta), r * np.sin(theta), np.zeros(num_points)), axis=1).astype(np.float32)
                circle_imgpts, _ = cv2.projectPoints(circle_3d, rvec_marker, tvec_marker, camera_matrix, dist_coeffs)
                circle_imgpts = np.int32(circle_imgpts).reshape(-1, 2)
                cv2.polylines(frame, [circle_imgpts], isClosed=True, color=(0,0,255), thickness=2)
                
                # Project "top" point (marker coordinate: (0, marker_size/2, 0))
                top_point_marker = np.array([[0, marker_size/2, 0]], dtype=np.float32)
                top_imgpts, _ = cv2.projectPoints(top_point_marker, rvec_marker, tvec_marker, camera_matrix, dist_coeffs)
                top_imgpts = np.int32(top_imgpts).reshape(-1, 2)
                top_pt = top_imgpts[0]
                cv2.arrowedLine(frame, tuple(np.int32(center)), tuple(top_pt), (255, 0, 255), 2)
                
                # Compute and display yaw (rotation about z-axis)
                R_marker, _ = cv2.Rodrigues(rvec_marker)
                R_marker_global = R_cam_inv @ R_marker
                euler_angles = rotationMatrixToEulerAngles(R_marker_global)  # Now yaw is global
                yaw_deg = np.degrees(euler_angles[2])

                yaw_text = f"Yaw: {yaw_deg:.1f} deg"
                cv2.putText(frame, yaw_text, tuple(np.int32(center + np.array([5,15]))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                # Put text with marker global position
                cv2.putText(frame, text, tuple(np.int32(center + np.array([5, -5]))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Marker ID {marker_id} global pos: {t_marker_world.flatten()} Yaw: {yaw_deg:.1f} deg")
                
                # Store the position and orientation data for JSON update
                markers_data[str(marker_id)] = {
                    "position": t_marker_world.flatten().tolist(),
                    "yaw_deg": yaw_deg
                }
            else:
                print(f"solvePnP failed for marker {marker_id}")

    # Write marker data to JSON file every frame
    with open("markers_positions.json", "w") as json_file:
        json.dump(markers_data, json_file)
    
    # Draw the Global Axes (Trident)
    imgpts_axes, _ = cv2.projectPoints(axes_points_world, rvec_cam, tvec_cam, camera_matrix, dist_coeffs)
    imgpts_axes = np.int32(imgpts_axes).reshape(-1, 2)
    cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[1]), (0, 0, 255), 2)
    cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[2]), (0, 255, 0), 2)
    cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[3]), (255, 0, 0), 2)
    cv2.circle(frame, tuple(imgpts_axes[0]), 3, (0, 0, 0), -1)

    cv2.imshow("Global 3D Markers & Trident", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
