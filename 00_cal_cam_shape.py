import cv2
import numpy as np
import json
import os
import time

# Set the camera index (adjust as needed)
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

# Define the checkerboard size (number of inner corners per checkerboard row and column)
checkerboard_size = (9, 6)  # (columns, rows)
square_size = 2.0  # Size of each square in your checkerboard in cm (or any unit)

# Prepare object points, like (0,0,0), (1,0,0), ... (8,5,0)
# Multiply by square_size to convert to real-world units
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images
obj_points = []  # 3d points in real world space
img_points = []  # 2d points in image plane

# Initialize capture count
count = 0
TOTAL_IMAGES = 250  # Total number of good frames to capture

# Create a full-screen window.
window_name = "Calibration of camera"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while count < TOTAL_IMAGES:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")

        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret_cb:
        # Add object points and image points if corners are found
        obj_points.append(objp)
        img_points.append(corners)

        # Draw the detected corners
        cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret_cb)

        count += 1
        print(f"Captured {count} good checkerboard images.")

    # Overlay feedback text on the image (e.g., "Captured: 12 / 250 images")
    feedback_text = f"Captured: {count} / {TOTAL_IMAGES} images"
    cv2.putText(frame, feedback_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display the image
    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(200)  # Delay of 0.2 seconds between frames
    if key == 27:  # Press 'Esc' to exit early
        break

cap.release()
cv2.destroyAllWindows()

if len(obj_points) >= 100:

    ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'distortion_coefficients': dist.tolist(),
        'rotation_vectors': [rvec.tolist() for rvec in rvecs],
        'translation_vectors': [tvec.tolist() for tvec in tvecs]
    }

    with open('camera_calibration_data.json', 'w') as json_file:
        json.dump(calibration_data, json_file, indent=4)

    print("Camera calibration data saved in 'camera_calibration_data.json'.")
else:
    print("Insufficient good checkerboard captures. Calibration failed.")
