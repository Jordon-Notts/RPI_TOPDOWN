import cv2
import numpy as np
import json

# ---------------------------
# Configuration and Loading
# ---------------------------

# Define the physical marker size (in your chosen units, e.g. mm)
marker_size = 100  # e.g., 100 mm

# Define the marker's 3D object points in its own coordinate system.
# Here we center the marker at the origin: corners are defined as:
# top-left: (-marker_size/2, marker_size/2, 0)
# top-right: (marker_size/2, marker_size/2, 0)
# bottom-right: (marker_size/2, -marker_size/2, 0)
# bottom-left: (-marker_size/2, -marker_size/2, 0)
marker_obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# Load camera calibration data (intrinsics)
with open('camera_calibration_data.json', 'r') as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)

# Load the camera pose (extrinsics) from the previous calibration.
# These were computed so that the camera coordinates relate to world coordinates via:
#    X_cam = R_cam * X_world + t_cam.
# Thus, to transform a point from camera to world, we compute:
#    X_world = R_cam^T * (X_cam - t_cam)
with open('camera_pose.json', 'r') as f:
    pose_data = json.load(f)
rvec_cam = np.array(pose_data["rotation_vector"], dtype=np.float32)
tvec_cam = np.array(pose_data["translation_vector"], dtype=np.float32)
R_cam, _ = cv2.Rodrigues(rvec_cam)  # rotation from world to camera

# For transforming from camera coordinates to world, compute the inverse rotation:
R_cam_inv = R_cam.T

# ---------------------------
# Initialize Webcam & Detector
# ---------------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Use the predefined ArUco dictionary (standard 4x4 markers from Arucogen)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()  # Use default parameters

print("Starting marker detection. Press 'Esc' to exit.")

# ---------------------------
# Main Loop: Detect and Compute 3D Position
# ---------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Draw detected markers (edges and IDs)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # For each detected marker, estimate its pose relative to the camera
        for i, marker_id in enumerate(ids.flatten()):
            # Get the 2D image points for this marker (should be 4 corners)
            img_pts = corners[i][0].astype(np.float32)  # shape (4,2)
            
            # Use solvePnP to get the marker pose relative to the camera.
            # This returns rvec_marker and tvec_marker such that:
            # X_cam_marker = R_marker * X_marker_obj + tvec_marker.
            ret_pnp, rvec_marker, tvec_marker = cv2.solvePnP(marker_obj_points, img_pts,
                                                             camera_matrix, dist_coeffs,
                                                             flags=cv2.SOLVEPNP_ITERATIVE)
            if ret_pnp:
                # Convert marker pose from camera to world coordinates.
                # The marker's position in camera coordinates is given by tvec_marker.
                # The world coordinates of the marker origin are:
                #    X_marker_world = R_cam_inv * (tvec_marker - tvec_cam)
                t_marker_world = R_cam_inv @ (tvec_marker - tvec_cam)
                
                # Optionally, convert the rotation of the marker (rvec_marker) to a rotation matrix,
                # then to world coordinates:
                R_marker, _ = cv2.Rodrigues(rvec_marker)
                R_marker_world = R_cam_inv @ R_marker  # not used for location, but for full pose

                # Draw the marker edges (already drawn by aruco.drawDetectedMarkers)
                # Overlay the computed 3D position (world coordinates) on the image.
                pos_text = f"ID {marker_id}: [{t_marker_world.flatten()[0]:.1f}, " \
                           f"{t_marker_world.flatten()[1]:.1f}, {t_marker_world.flatten()[2]:.1f}]"
                # Place text near the top-left corner of the marker
                cv2.putText(frame, pos_text, tuple(np.int32(img_pts[0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                print(f"Marker ID {marker_id} world position: {t_marker_world.flatten()}")
                
    cv2.imshow("Webcam - Marker 3D Position", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Esc to exit
        break

cap.release()
cv2.destroyAllWindows()