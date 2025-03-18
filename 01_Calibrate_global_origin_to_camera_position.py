import cv2
import numpy as np
import json

# ----- CONFIGURATION -----
# Known cube side length (for example, in mm)
cube_size = 250

# Define the world (3D) coordinates for all 8 vertices of the cube.
# The cube is assumed to have one vertex at the origin and edges along the positive axes.
object_points_dict = {
    "0": np.array([0, 0, 0], dtype=np.float32),                     # Vertex 0: Origin (bottom)
    "1": np.array([cube_size, 0, 0], dtype=np.float32),               # Vertex 1: X bottom front
    "2": np.array([cube_size, cube_size, 0], dtype=np.float32),        # Vertex 2: Furthest from origin (bottom)
    "3": np.array([0, cube_size, 0], dtype=np.float32),               # Vertex 3: Y bottom front
    "4": np.array([0, 0, cube_size], dtype=np.float32),               # Vertex 4: Origin (top)
    "5": np.array([cube_size, 0, cube_size], dtype=np.float32),         # Vertex 5: X top front
    "6": np.array([cube_size, cube_size, cube_size], dtype=np.float32),  # Vertex 6: Furthest from origin (top)
    "7": np.array([0, cube_size, cube_size], dtype=np.float32)          # Vertex 7: Y top front
}

# ----- LOAD CAMERA CALIBRATION DATA (INTRINSICS) -----
with open('camera_calibration_data.json', 'r') as f:
    calib_data = json.load(f)
camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)

# Global dictionary to store selected image points.
# Keys will be strings "0" to "7" corresponding to the cube vertices.
image_points_dict = {}

# Current label to be selected (set by key press).
current_label = None

instructions = (
    "Press keys 0-7 to select a cube vertex:\n"
    "0: (0,0,0) – Origin (bottom)\n"
    "1: (L,0,0) – X bottom front\n"
    "2: (L,L,0) – Furthest from origin (bottom)\n"
    "3: (0,L,0) – Y bottom front\n"
    "4: (0,0,L) – Origin (top)\n"
    "5: (L,0,L) – X top front\n"
    "6: (L,L,L) – Furthest from origin (top)\n"
    "7: (0,L,L) – Y top front\n"
    "Then click on the image to mark the point.\n"
    "Select at least 6 points.\n"
    "Press 'p' to perform pose estimation,\n"
    "'r' to reset selections,\n"
    "'s' to save the pose,\n"
    "and Esc to exit."
)

# Mouse callback: When a click occurs and current_label is set, store the point.
def mouse_callback(event, x, y, flags, param):
    global current_label, image_points_dict
    if event == cv2.EVENT_LBUTTONDOWN and current_label is not None:
        image_points_dict[current_label] = (x, y)
        print(f"Point for vertex '{current_label}' set to: ({x}, {y})")

# Create window and set mouse callback.

window_name = "Cube Pose Estimation"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

computed_pose = None  # Will store the result from solvePnP

# Define cube vertices for drawing the cube (all 8 vertices) – same order as in object_points_dict.
cube_vertices = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, cube_size],
    [cube_size, 0, cube_size],
    [cube_size, cube_size, cube_size],
    [0, cube_size, cube_size]
], dtype=np.float32)

# Define the cube edges (pairs of indices into cube_vertices)
cube_edges = [
    (0,1), (1,2), (2,3), (3,0),  # bottom face
    (4,5), (5,6), (6,7), (7,4),  # top face
    (0,4), (1,5), (2,6), (3,7)   # vertical edges
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Draw the selected points.
    for label, pt in image_points_dict.items():
        cv2.circle(frame, pt, 1, (0, 255, 255), -1)
        cv2.putText(frame, label, (pt[0]+5, pt[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Display instructions.
    y0, dy = 20, 20
    for i, line in enumerate(instructions.split("\n")):
        y = y0 + i*dy
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # If pose has been computed, overlay the trident and cube.
    if computed_pose is not None:
        rvec, tvec = computed_pose

        # --- Draw Trident ---
        axes_obj_points = np.array([
            [0, 0, 0],                     # origin
            [cube_size, 0, 0],             # X-axis endpoint (red)
            [0, cube_size, 0],             # Y-axis endpoint (green)
            [0, 0, cube_size]              # Z-axis endpoint (blue)
        ], dtype=np.float32)
        imgpts_axes, _ = cv2.projectPoints(axes_obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts_axes = np.int32(imgpts_axes).reshape(-1, 2)
        cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[1]), (0,0,255), 2)
        cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[2]), (0,255,0), 2)
        cv2.arrowedLine(frame, tuple(imgpts_axes[0]), tuple(imgpts_axes[3]), (255,0,0), 2)
        cv2.circle(frame, tuple(imgpts_axes[0]), 5, (0,0,0), -1)

        # --- Draw Cube (wireframe) ---
        imgpts_cube, _ = cv2.projectPoints(cube_vertices, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts_cube = np.int32(imgpts_cube).reshape(-1, 2)
        for edge in cube_edges:
            pt1 = tuple(imgpts_cube[edge[0]])
            pt2 = tuple(imgpts_cube[edge[1]])
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)  # cyan edges

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    # Set current label based on number key presses (keys '0' to '7').
    if key in [ord(str(i)) for i in range(0,8)]:
        current_label = chr(key)
        print(f"Select vertex {current_label}.")
    elif key == ord('r'):
        image_points_dict = {}
        computed_pose = None
        print("Selections reset.")
    # When 'p' is pressed, perform pose estimation if at least 6 points are selected.
    elif key == ord('p'):
        if len(image_points_dict) < 6:
            print("At least 6 points are required. Currently selected:", len(image_points_dict))
        else:
            # Use the selected points; sort by key (numerical order) for consistency.
            selected_keys = sorted(image_points_dict.keys(), key=lambda k: int(k))
            img_pts = []
            obj_pts = []
            for key_label in selected_keys:
                img_pts.append(image_points_dict[key_label])
                obj_pts.append(object_points_dict[key_label])
            img_pts = np.array(img_pts, dtype=np.float32)
            obj_pts = np.array(obj_pts, dtype=np.float32)
            ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if ret:
                computed_pose = (rvec, tvec)
                print("Pose computed:")
                print("Rotation Vector:", rvec.flatten())
                print("Translation Vector:", tvec.flatten())
            else:
                print("Pose estimation failed.")
    # Save the computed pose.
    elif key == ord('s'):
        if computed_pose is not None:
            new_pose = {
                "rotation_vector": computed_pose[0].tolist(),
                "translation_vector": computed_pose[1].tolist()
            }
            with open("camera_pose.json", "w") as f:
                json.dump(new_pose, f, indent=4)
            print("New camera pose saved to 'camera_pose.json':", new_pose)
        else:
            print("No pose computed to save.")
    elif key == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()
