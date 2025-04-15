#!/bin/bash
# Activate the virtual environment (adjust the path as needed)
source ./venv/bin/activate

# Run the Flask server in the foreground
python3 07_Share_position_data_on_network.py &

# Run the marker detection script in the background
python3 02_Get_Tag_Pos_and_Store_in_JSON.py