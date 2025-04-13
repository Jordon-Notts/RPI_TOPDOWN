#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import time
import math
from datetime import datetime

# -----------------------------------------------------------
# Configuration / Paths and Parameters
# -----------------------------------------------------------
CAMERA_INDEX = 2
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
marker_size = 100           # in mm (adjust as needed)
CAPTURE_ITERATIONS = 10     # Number of frames to average when capturing

# -----------------------------------------------------------
# Load Camera Calibration Data
# -----------------------------------------------------------
CALIBRATION_FILE = "camera_calibration_data.json"
if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(f"Missing {CALIBRATION_FILE}")
with open(CALIBRATION_FILE, "r") as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
dist_coeffs   = np.array(calib_data["distortion_coefficients"], dtype=np.float32)

# -----------------------------------------------------------
# Load Camera Pose (global transformation)
# -----------------------------------------------------------
CAMERA_POSE_FILE = "camera_pose.json"
if not os.path.exists(CAMERA_POSE_FILE):
    raise FileNotFoundError(f"Missing {CAMERA_POSE_FILE}")
with open(CAMERA_POSE_FILE, "r") as f:
    pose_data = json.load(f)
rvec_cam = np.array(pose_data["rotation_vector"], dtype=np.float32)
tvec_cam = np.array(pose_data["translation_vector"], dtype=np.float32)
R_cam, _ = cv2.Rodrigues(rvec_cam)
R_cam_inv = R_cam.T  # Inverse of camera rotation matrix

# -----------------------------------------------------------
# Load Marker Database (for filtering markers)
# -----------------------------------------------------------
MARKER_DB_FILE = "aruco_marker_positions.json"
try:
    with open(MARKER_DB_FILE, "r") as f:
        marker_db = json.load(f)
except Exception as e:
    print(f"Warning: Could not load {MARKER_DB_FILE} (error: {e}). Proceeding with an empty marker_db.")
    marker_db = {}

# -----------------------------------------------------------
# Define Marker Model (object points, in mm)
# -----------------------------------------------------------
marker_obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# -----------------------------------------------------------
# Setup ArUco Detector
# -----------------------------------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# -----------------------------------------------------------
# Initialize Webcam Capture
# -----------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1820)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Create a full-screen window.
window_name = "Robot Marker Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# -----------------------------------------------------------
# Global Variables for Capture Mode and Persistent Overlay
# -----------------------------------------------------------
persistent_markers = {}  # Format: { marker_id: { "corner_points": {corner_name: {x,y,z} } } }
capturing = False        # Flag indicating capture mode
capture_count = 0        # Number of frames captured during current capture session
accumulated_data = {}    # Accumulates subpixel refined 2D corner positions

def rotationMatrixToEulerAngles(R):
    """
    Converts a rotation matrix to Euler angles (roll, pitch, yaw).
    """
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0
    return np.array([roll, pitch, yaw])

def process_and_display():
    """
    Continuously captures frames and displays live video with overlays:
      - Live marker detections.
      - Global origin.
      - Instructions.
      - Persistent marker data (from the last capture, if available).
      
    When 'c' is pressed, the script enters capture mode, accumulating refined marker corner
    positions over CAPTURE_ITERATIONS frames. Then it computes the average image corner for
    each marker, uses solvePnP with the averaged 2D points to compute global coordinates (via
    the camera pose), updates the JSON file ("aruco_marker_positions.json") and stores the results 
    in persistent_markers so the data remains overlaid.
      
    Press Esc to exit.
    """
    global capturing, capture_count, accumulated_data, persistent_markers, rvec_cam, tvec_cam

    instructions = ("Marker Mode\n"
                    " - Detected markers will be drawn\n"
                    " - Press 'c' to capture & persist tag positions\n"
                    " - Press Esc to exit")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        # Draw live detected markers.
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Overlay persistent marker data, if available.
        if persistent_markers:
            y_text = 100
            for marker_id, data in persistent_markers.items():
                txt = f"ID {marker_id}: "
                for cname, coords in data["corner_points"].items():
                    txt += f"{cname}=({coords['x']:.1f},{coords['y']:.1f},{coords['z']:.1f})  "
                cv2.putText(frame, txt, (20, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_text += 20

        # Always project and draw the global origin.
        origin_world = np.array([[0, 0, 0]], dtype=np.float32)
        origin_imgpts, _ = cv2.projectPoints(origin_world, rvec_cam, tvec_cam, camera_matrix, dist_coeffs)
        origin_pt = np.int32(origin_imgpts).reshape(-1, 2)[0]
        cv2.circle(frame, tuple(origin_pt), 8, (0, 255, 255), -1)
        cv2.putText(frame, "Origin", (origin_pt[0] + 10, origin_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Overlay instructions.
        y0 = 20
        for i, line in enumerate(instructions.split("\n")):
            cv2.putText(frame, line, (10, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Process key press events (use one single call per frame).
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key: exit.
            break
        elif key == ord('c'):
            if not capturing:
                capturing = True
                capture_count = 0
                accumulated_data = {}
                print("Starting capture mode for averaging marker corners.")

        # If capturing, accumulate refined corners.
        if capturing:
            if ids is not None and len(ids) > 0:
                for i, marker in enumerate(corners):
                    marker_id = str(ids[i][0])
                    # if marker_id not in marker_db:
                    #     continue
                    refined = cv2.cornerSubPix(gray, marker, (3,3), (-1,-1), criteria)
                    pts = refined.reshape(-1, 2)
                    # Draw refined corners for feedback.
                    for pt in pts:
                        cv2.circle(frame, tuple(np.int32(pt)), 5, (0, 255, 0), -1)
                    # Accumulate corners.
                    if marker_id not in accumulated_data:
                        accumulated_data[marker_id] = {
                            "top_left": [],
                            "top_right": [],
                            "bottom_right": [],
                            "bottom_left": []
                        }
                    for idx, cname in enumerate(["top_left", "top_right", "bottom_right", "bottom_left"]):
                        accumulated_data[marker_id][cname].append(pts[idx].tolist())
            capture_count += 1
            cv2.putText(frame, f"Capturing... {capture_count}/{CAPTURE_ITERATIONS}", 
                        (20, frame.shape[0]-80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Check if capture averaging is complete
            if capture_count >= CAPTURE_ITERATIONS:
                new_markers = {} # This will hold the results of THIS capture session
                for marker_id, corners_dict in accumulated_data.items():
                    averaged_corners = {}
                    for cname, values in corners_dict.items():
                        if len(values) > 0:
                            avg_coords = np.mean(np.array(values), axis=0)
                            averaged_corners[cname] = {"x": float(avg_coords[0]),
                                                        "y": float(avg_coords[1])}
                    order = ["top_left", "top_right", "bottom_right", "bottom_left"]
                    avg_img_pts = []
                    # Ensure all corners were found before proceeding
                    valid_avg = True
                    for cname in order:
                        if cname not in averaged_corners:
                            print(f"Warning: Missing averaged corner '{cname}' for marker {marker_id}. Skipping solvePnP.")
                            valid_avg = False
                            break
                        pt = averaged_corners[cname]
                        avg_img_pts.append([pt["x"], pt["y"]])

                    if not valid_avg:
                        continue # Skip this marker if averaging failed for any corner

                    avg_img_pts = np.array(avg_img_pts, dtype=np.float32)
                    ret_pnp, rvec_marker, tvec_marker = cv2.solvePnP(marker_obj_points, avg_img_pts,
                                                                      camera_matrix, dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)
                    if ret_pnp:
                        # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_marker, tvec_marker, 50) # Optional: Draw axes on live frame
                        R_marker, _ = cv2.Rodrigues(rvec_marker)
                        global_corners = {}
                        for idx, cname in enumerate(order):
                            p_model = marker_obj_points[idx].reshape((3,1))
                            p_cam = R_marker.dot(p_model) + tvec_marker
                            p_global = R_cam_inv.dot(p_cam - tvec_cam)
                            global_corners[cname] = {"x": float(p_global[0,0]),
                                                      "y": float(p_global[1,0]),
                                                      "z": float(p_global[2,0])}
                            # Optional: Draw calculated global coords on live frame
                            # pt_avg = np.mean(avg_img_pts, axis=0)
                            # txt = f"{cname}: ({p_global[0,0]:.1f},{p_global[1,0]:.1f},{p_global[2,0]:.1f})"
                            # cv2.putText(frame, txt, (int(pt_avg[0])+5, int(pt_avg[1])-5),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        new_markers[marker_id] = {"corner_points": global_corners}
                        print(f"Marker {marker_id} global corners (averaged): {global_corners}")
                    else:
                        print(f"solvePnP failed for marker {marker_id} during capture averaging.")

                # --- Load existing data, merge, and save ---
                output_json = os.path.join(os.getcwd(), MARKER_DB_FILE) # Use constant
                existing_data = {}

                # 1. Load existing data from JSON
                try:
                    if os.path.exists(output_json):
                        with open(output_json, "r") as f_read:
                            existing_data = json.load(f_read)
                            # Basic validation: Ensure it's a dictionary
                            if not isinstance(existing_data, dict):
                                print(f"Warning: {output_json} does not contain a valid JSON dictionary object. Starting fresh.")
                                existing_data = {}
                    # else: File doesn't exist, existing_data remains {} which is correct.

                except json.JSONDecodeError:
                    print(f"Warning: {output_json} contains invalid JSON. Starting fresh.")
                    existing_data = {} # Reset to empty if file is corrupt
                except Exception as e:
                    print(f"Error reading {output_json}: {e}. Starting fresh.")
                    existing_data = {} # Reset to empty on other read errors

                # 2. Merge the newly captured data into the existing data
                #    .update() handles both adding new keys and updating existing ones.
                existing_data.update(new_markers)

                # 3. Save the combined (merged) data back to the file
                try:
                    with open(output_json, "w") as f_write:
                        json.dump(existing_data, f_write, indent=4)
                    print(f"Marker positions updated in {output_json}")
                except Exception as e:
                    print(f"Error writing updated data to {output_json}: {e}")

                # 4. Update persistent display with the full merged data
                persistent_markers = existing_data.copy()

                # Reset capture state
                capturing = False
                capture_count = 0
                accumulated_data = {} # Reset accumulator

        cv2.imshow(window_name, frame)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_and_display()
