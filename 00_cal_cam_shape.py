import cv2
import numpy as np
import json
import time

# Define the checkerboard size (number of inner corners per checkerboard row and column)
checkerboard_size = (9, 6)  # (columns, rows)
square_size = 2.0  # Size of each square in your checkerboard in cm (or any unit)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
# Multiply by square_size to convert to real-world units
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by the square size to convert to real-world units

# Arrays to store object points and image points from all the images
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane

# Open the webcam
cap = cv2.VideoCapture(2)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize capture count
count = 0

# Run the calibration process, capturing until 100 good frames are obtained
while count < 100:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the image to grayscale for detecting the checkerboard
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        # If corners found, add object points and image points
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)

        count += 1
        print(f"Captured {count} good checkerboard images.")

    # Display the image with the detected checkerboard corners
    cv2.imshow('Checkerboard Calibration', frame)

    # Wait for 0.2 seconds before capturing the next frame
    key = cv2.waitKey(200)  # 0.2s delay
    
    if key == 27:  # Press 'Esc' to exit the loop early
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Perform camera calibration if sufficient captures are obtained
if len(obj_points) >= 100:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Save the calibration data to a dictionary
    calibration_data = {
        'camera_matrix': mtx.tolist(),  # Convert numpy arrays to lists for JSON
        'distortion_coefficients': dist.tolist(),
        'rotation_vectors': [rvec.tolist() for rvec in rvecs],  # Convert each rotation vector to a list
        'translation_vectors': [tvec.tolist() for tvec in tvecs]  # Convert each translation vector to a list
    }

    # Save the calibration data in a JSON file
    with open('camera_calibration_data.json', 'w') as json_file:
        json.dump(calibration_data, json_file, indent=4)

    print("Camera calibration data saved in 'camera_calibration_data.json'.")
else:
    print("Insufficient good checkerboard captures. Calibration failed.")
