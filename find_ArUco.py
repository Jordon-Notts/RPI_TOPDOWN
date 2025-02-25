import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Use the correct dictionary for standard 4x4 markers.
# For standard Arucogen 4x4 markers, the dictionary is usually DICT_4X4_50,
# and the marker size (in bits) should be 4.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert the image to grayscale for ArUco marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        for i, corner in enumerate(corners):
            # Draw the edges of the detected ArUco marker
            cv2.polylines(frame, [np.int32(corner)], isClosed=True, color=(0, 255, 0), thickness=2)
            # Annotate the marker with its ID
            cv2.putText(frame, str(ids[i][0]), tuple(np.int32(corner[0][0])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the webcam feed with marked ArUco tags
    cv2.imshow('Webcam Stream with ArUco Edges', frame)

    # Wait for key press and exit if 'Esc' is pressed
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
