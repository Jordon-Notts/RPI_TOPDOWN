from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

def load_markers_data():
    json_filename = "markers_positions.json"
    if os.path.exists(json_filename):
        try:
            with open(json_filename, "r") as f:
                markers_data = json.load(f)
            return markers_data
        except Exception as e:
            return {}
    else:
        return {}

@app.route('/')
def index():
    markers_data = load_markers_data()
    links = ""
    if markers_data:
        for marker_id in markers_data.keys():
            links += f'<li><a href="/robot/{marker_id}">Robot {marker_id}</a></li>'
    else:
        links = "<li>No markers available at this time.</li>"
    
    html = f'''
    <html>
        <head>
            <title>Markers Index</title>
        </head>
        <body>
            <h1>Available Marker Routes</h1>
            <ul>
                {links}
            </ul>
        </body>
    </html>
    '''
    return html

@app.route('/robot/<robot_number>')
def robot(robot_number):
    markers_data = load_markers_data()
    if robot_number in markers_data:
        return jsonify(markers_data[robot_number])
    else:
        return jsonify({"error": f"Marker with id {robot_number} not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
