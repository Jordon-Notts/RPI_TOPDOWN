# Top down Aruco tag positioning system

This code is used to find aruco tags in the webcam stream, and returns there position.

# Install

It is expected this this is run on a inux system

```bash

mkdir TOP_DOWN_POSTIONING_SYSTEM

cd TOP_DOWN_POSTIONING_SYSTEM

wget https://github.com/Jordon-Notts/RPI_TOPDOWN/archive/refs/heads/main.zip

unzip main.zip

cp RPI_TOPDOWN-main/* .

rm main.zip

rm RPI_TOPDOWN-main/ -d -r

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

```

# Setup

## Calibrate camera lens

The camera shape needs to be calibrated before any meaningful data can be obtained (ref https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). To calibrate the camera first print the calibration pattern https://github.com/opencv/opencv/blob/4.x/doc/pattern.png. You will need to measure the size of the squares and update the parameeter in the script (line 7 00_cal_cam_shape.py)

```python

square_size = 2.0  # Size of each square in your checkerboard in cm (or any unit)

```

```bash

nano 00_cal_cam_shape.py

```

Then control-x to save and close.

Then run the script.

```python

python3 00_cal_cam_shape.py

```

Assuming everythig works ok, then hold the chequer board up to the camera and move it around. 100 images will be taken which will allows the distortion of the lens and focal lenght to be recorded and acounted for.

### Possible issues

if the computor you have has an in build webcam, and for this you are using an external you should be ok using the value 2. if you dont have an internal webcam and you are using an external then a value if 0 should work. if not try a series of differnt numbers.

line 21 00_cal_cam_shape.py

```python

# Open the webcam
cap = cv2.VideoCapture(2)

```

## Calibrate global space

This part of the setup is to tell the system where the global origin is.

For this i used an external webcam and a tripod, a long usb cable and a cube. The tripod was set up so the entire scene can be seen.

The cube has to have equal sides, in order to change the code for different size cube, change this line (Line 7).

```python 

# ----- CONFIGURATION -----
# Known cube side length (for example, in mm)
cube_size = 250

```

Run the script and identify the corners of the cubes.

Press p to calculte the pose of the scene, this will show a cube over the image.

if you are happpy with the set up, press s to save.

# Run

Downaload and print the markers from https://chev.me/arucogen/

ensure the correct size of the markers are inputted into the script (line 7)

```python

marker_size = 232  # adjust as needed

```

to run the scripts type:

```bash

bash Run_GPS_Server.sh 

```


