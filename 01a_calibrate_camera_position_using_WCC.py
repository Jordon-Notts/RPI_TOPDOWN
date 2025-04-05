import cv2
import numpy as np
import json
import os
from datetime import datetime

# ----- Load Camera Calibration Data -----
with open('camera_calibration_data.json', 'r') as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)

# ----- Cube Calibration Configuration -----
cube_size = 250  # known cube side length in mm
# Define base cube vertices (at the origin)
object_points_dict = {
    "0": np.array([0, 0, 0], dtype=np.float32),
    "1": np.array([cube_size, 0, 0], dtype=np.float32),
    "2": np.array([cube_size, cube_size, 0], dtype=np.float32),
    "3": np.array([0, cube_size, 0], dtype=np.float32),
    "4": np.array([0, 0, cube_size], dtype=np.float32),
    "5": np.array([cube_size, 0, cube_size], dtype=np.float32),
    "6": np.array([cube_size, cube_size, cube_size], dtype=np.float32),
    "7": np.array([0, cube_size, cube_size], dtype=np.float32)
}
# For cube calibration, store current unsaved 2D image points.
image_points_dict = {}
current_label = None  # The currently selected cube vertex

# Global list to accumulate multiple cube calibration sets.
# Each set is a tuple: (object_points, image_points, offset)
cube_calibration_sets = []
# Current cube offset (to be added to base cube vertices)
cube_offset = np.array([0, 0, 0], dtype=np.float32)

# Folder to save annotated images.
SAVE_FOLDER = "camera_localization_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Global variable to hold the latest computed pose.
computed_pose = None  # (rvec, tvec)

# ----- Mouse Callback for Cube Mode -----
def mouse_callback(event, x, y, flags, param):
    global current_label, image_points_dict
    if event == cv2.EVENT_LBUTTONDOWN and current_label is not None:
        image_points_dict[current_label] = (x, y)
        print(f"Set vertex {current_label} at image point ({x}, {y})")

# ----- Setup Video Capture and Window -----
camera_index = 2  # Adjust camera index as needed
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

window_name = "Cube Calibration Mode"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

# ----- Instructions -----
instructions = (
    "Cube Calibration Mode\n"
    " - Press keys 0-7 to select a cube vertex and click to mark its image point.\n"
    " - At least 6 points are needed per cube.\n"
    " - Offset adjustments:\n"
    "       'x': increase X by 100mm\n"
    "       'c': decrease X by 100mm\n"
    "       'y': increase Y by 100mm\n"
    "       'u': decrease Y by 100mm\n"
    " - Press 'a' to append current cube to calibration sets.\n"
    " - Press 'p' to compute pose using all appended cubes (+ current unsaved if valid).\n"
    " - Press 's' to save the computed pose and annotated image.\n"
    " - Press 'r' to reset current unsaved points.\n"
    " - Press Esc to exit."
)

# Define full cube for drawing edges and corners.
base_cube = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, cube_size],
    [cube_size, 0, cube_size],
    [cube_size, cube_size, cube_size],
    [0, cube_size, cube_size]
], dtype=np.float32)
cube_edges = [(0,1), (1,2), (2,3), (3,0),
              (4,5), (5,6), (6,7), (7,4),
              (0,4), (1,5), (2,6), (3,7)]

# ----- Main Loop -----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # --- Top Left: Cube Offset Text ---
    offset_text = f"Cube Offset: X={cube_offset[0]}, Y={cube_offset[1]}, Z={cube_offset[2]}"
    cv2.putText(frame, offset_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    # --- Below Offset: Instructions ---
    y0 = 50
    for i, line in enumerate(instructions.split("\n")):
        cv2.putText(frame, line, (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Draw unsaved cube points.
    for label, pt in image_points_dict.items():
        cv2.circle(frame, pt, 4, (0,255,255), -1)
        cv2.putText(frame, label, (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    # Draw saved calibration sets (displayed in red).
    for set_idx, (obj_pts_set, img_pts_set, offset) in enumerate(cube_calibration_sets):
        for pt in img_pts_set:
            cv2.circle(frame, tuple(pt.astype(int)), 4, (0,0,255), -1)
        cv2.putText(frame, f"Set {set_idx+1}", tuple(img_pts_set[0].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # If a pose has been computed, continuously overlay the pose and cube projections.
    if computed_pose is not None:
        rvec, tvec = computed_pose
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)
        # Compute camera world position and heading.
        R, _ = cv2.Rodrigues(rvec)
        cam_pos = -R.T.dot(tvec)
        cam_pos_flat = cam_pos.flatten()
        forward_vector = R.T.dot(np.array([[0], [0], [1]], dtype=np.float32))
        heading_angle = np.degrees(np.arctan2(forward_vector[1,0], forward_vector[0,0]))
        if heading_angle < 0:
            heading_angle += 360
        # Bottom of frame: Camera position and heading.
        pos_text = f"Cam Pos: X={cam_pos_flat[0]:.1f}, Y={cam_pos_flat[1]:.1f}, Z={cam_pos_flat[2]:.1f}"
        heading_text = f"Heading: {heading_angle:.1f} deg"
        cv2.putText(frame, pos_text, (20, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, heading_text, (20, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        # Draw cube edges and corners for each saved calibration set (green).
        for (obj_pts_set, img_pts_set, offset) in cube_calibration_sets:
            cube_world = base_cube + offset
            imgpts_cube, _ = cv2.projectPoints(cube_world, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts_cube = np.int32(imgpts_cube).reshape(-1,2)
            for edge in cube_edges:
                pt1 = tuple(imgpts_cube[edge[0]])
                pt2 = tuple(imgpts_cube[edge[1]])
                cv2.line(frame, pt1, pt2, (0,255,0), 2)
            for pt in imgpts_cube:
                cv2.circle(frame, tuple(pt), 5, (0,255,0), -1)
        # Draw edges and corners for current unsaved cube (magenta).
        if len(image_points_dict) >= 6:
            cube_world = base_cube + cube_offset
            imgpts_cube, _ = cv2.projectPoints(cube_world, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts_cube = np.int32(imgpts_cube).reshape(-1,2)
            for edge in cube_edges:
                pt1 = tuple(imgpts_cube[edge[0]])
                pt2 = tuple(imgpts_cube[edge[1]])
                cv2.line(frame, pt1, pt2, (255,0,255), 2)
            for pt in imgpts_cube:
                cv2.circle(frame, tuple(pt), 5, (255,0,255), -1)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    # --- Key Commands ---
    if key == 27:  # Esc key
        break
    elif key in [ord(str(i)) for i in range(8)]:
        current_label = chr(key)
        print(f"Current vertex set to {current_label}")
    elif key == ord('r'):
        image_points_dict = {}
        print("Unsaved cube points reset.")
    elif key == ord('x'):
        cube_offset[0] += 100
        print("Cube offset increased in X by 100mm")
    elif key == ord('c'):
        cube_offset[0] -= 100
        print("Cube offset decreased in X by 100mm")
    elif key == ord('y'):
        cube_offset[1] += 100
        print("Cube offset increased in Y by 100mm")
    elif key == ord('u'):
        cube_offset[1] -= 100
        print("Cube offset decreased in Y by 100mm")
    elif key == ord('a'):
        if len(image_points_dict) < 6:
            print("Not enough points to append. At least 6 needed.")
        else:
            selected_keys = sorted(image_points_dict.keys(), key=lambda k: int(k))
            curr_obj_points = []
            curr_img_points = []
            for key_label in selected_keys:
                if key_label in object_points_dict:
                    pt_world = object_points_dict[key_label] + cube_offset
                    curr_obj_points.append(pt_world)
                    curr_img_points.append(image_points_dict[key_label])
            curr_obj_points = np.array(curr_obj_points, dtype=np.float32)
            curr_img_points = np.array(curr_img_points, dtype=np.float32)
            cube_calibration_sets.append((curr_obj_points, curr_img_points, cube_offset.copy()))
            print(f"Appended calibration set with offset {cube_offset.tolist()}")
            image_points_dict = {}  # Reset unsaved points.
    elif key == ord('p'):
        # Pose estimation: Combine all saved sets plus unsaved (if valid).
        all_obj_points_list = []
        all_img_points_list = []
        for (obj_pts_set, img_pts_set, offset) in cube_calibration_sets:
            all_obj_points_list.append(obj_pts_set)
            all_img_points_list.append(img_pts_set)
        if len(image_points_dict) >= 6:
            selected_keys = sorted(image_points_dict.keys(), key=lambda k: int(k))
            curr_obj_points = []
            curr_img_points = []
            for key_label in selected_keys:
                if key_label in object_points_dict:
                    pt_world = object_points_dict[key_label] + cube_offset
                    curr_obj_points.append(pt_world)
                    curr_img_points.append(image_points_dict[key_label])
            curr_obj_points = np.array(curr_obj_points, dtype=np.float32)
            curr_img_points = np.array(curr_img_points, dtype=np.float32)
            all_obj_points_list.append(curr_obj_points)
            all_img_points_list.append(curr_img_points)
        if len(all_obj_points_list) == 0:
            print("No calibration sets available for pose estimation.")
        else:
            all_obj_points = np.vstack(all_obj_points_list)
            all_img_points = np.vstack(all_img_points_list)
            ret, rvec, tvec = cv2.solvePnP(all_obj_points, all_img_points,
                                           camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            if ret:
                print("Pose estimated:")
                print("Rotation Vector:", rvec.flatten())
                print("Translation Vector:", tvec.flatten())
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 50)
                computed_pose = (rvec, tvec)
            else:
                print("solvePnP failed.")
    elif key == ord('s'):
        if computed_pose is not None:
            new_pose = {
                "rotation_vector": computed_pose[0].tolist(),
                "translation_vector": computed_pose[1].tolist()
            }
            with open("camera_pose.json", "w") as f:
                json.dump(new_pose, f, indent=4)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_out_path = os.path.join(SAVE_FOLDER, f"camera_pose_{ts}.jpg")
            cv2.imwrite(image_out_path, frame)
            print("Pose saved to 'camera_pose.json' and annotated image saved as", image_out_path)
        else:
            print("No pose computed to save.")

cap.release()
cv2.destroyAllWindows()
