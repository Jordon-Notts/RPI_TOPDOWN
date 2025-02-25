import cv2
import numpy as np
import json

# === Step 1: Load World Coordinates for Markers ===
# tag_locations.json should be like:
# {
#    "0": {"x": 0, "y": 500, "z": -600},
#    "1": {"x": 1400, "y": 70, "z": 0.0},
#    "6": {"x": 1400, "y": 500, "z": 0.0}
# }
with open('tag_locations.json', 'r') as f:
    tag_locations = json.load(f)

# Convert keys to integers and store world points in a dict
world_points = {}
for key, val in tag_locations.items():
    world_points[int(key)] = [val['x'], val['y'], val['z']]

# === Step 2: Load Camera Intrinsics (Calibration Data) ===
# camera_calibration_data.json should contain keys "camera_matrix" and "distortion_coefficients"
with open('camera_calibration_data.json', 'r') as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)

# === Step 3: Initialize Webcam and ArUco Detector ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Using predefined dictionary for standard 4x4 markers from Arucogen
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# We'll accumulate correspondences: for each detected marker (that is in our world_points),
# we compute its center in the image.
obj_points = []  # 3D world coordinates
img_points = []  # 2D image coordinates

print("Point your camera at the markers. Press 'c' to capture for calibration, or 'Esc' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        # Draw marker edges and IDs
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in world_points:
                # Compute center of marker: average of its 4 corners.
                center = np.mean(corners[i][0], axis=0)  # corners[i] shape: (1,4,2)
                # Draw the center as a small blue circle
                cv2.circle(frame, tuple(center.astype(int)), 5, (255, 0, 0), -1)
                # Annotate with marker id
                cv2.putText(frame, str(marker_id), tuple(center.astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Webcam Stream", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc to exit
        print("Exiting without calibration.")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == ord('c'):
        # On pressing 'c', capture correspondences from the current frame.
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in world_points:
                    center = np.mean(corners[i][0], axis=0)
                    img_points.append(center)
                    obj_points.append(world_points[marker_id])
            print(f"Captured {len(obj_points)} correspondences.")
            # If we have at least 3 correspondences, we can perform pose estimation.
            if len(obj_points) >= 3:
                break
            else:
                print("Not enough markers detected for calibration. Please try again.")

cap.release()
cv2.destroyAllWindows()

if len(obj_points) < 3:
    print("Calibration aborted: insufficient correspondences.")
    exit()

# Convert lists to numpy arrays (ensure shape: (N,3) and (N,2))
obj_points = np.array(obj_points, dtype=np.float32)
img_points = np.array(img_points, dtype=np.float32)

# Check how many correspondences we have:
print("Number of correspondences:", len(obj_points))

# Use a minimal solver if we have exactly 3 correspondences
if len(obj_points) == 3:
    ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
else:
    ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

if not ret:
    print("Pose estimation failed.")
    exit()

print("Estimated Rotation Vector (rvec):\n", rvec)
print("Estimated Translation Vector (tvec):\n", tvec)

import json

pose_data = {
    "rotation_vector": rvec.tolist(),
    "translation_vector": tvec.tolist()
}
with open("camera_pose.json", "w") as f:
    json.dump(pose_data, f, indent=4)

print("Camera pose saved to 'camera_pose.json'.")
