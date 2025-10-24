from flask import Flask, render_template, jsonify, request
from geopy.geocoders import Nominatim
from ml_logic import get_safe_paths  # ✅ Importing new ML logic

app = Flask(__name__)

GRID_SIZE = 50
STEPS = 20

# Sample drones data
drones_data = [
    {"name": "Drone 1", "latitude": 12.9716, "longitude": 77.5946, "altitude": 100},
    {"name": "Drone 2", "latitude": 12.9352, "longitude": 77.6245, "altitude": 0},
    {"name": "Drone 3", "latitude": 12.9279, "longitude": 77.6271, "altitude": 80},
]

# --- Utility functions ---
def get_coordinates_from_input(place_name):
    geolocator = Nominatim(user_agent="drone_navigation_app")
    location = geolocator.geocode(place_name + ", Bengaluru, India")
    if location:
        return (location.latitude, location.longitude)
    return None

def coord_to_grid(coord):
    lat_norm = (coord[0] - 12.8) / (13.1 - 12.8)
    lon_norm = (coord[1] - 77.4) / (77.8 - 77.4)
    return int(lat_norm * GRID_SIZE), int(lon_norm * GRID_SIZE)

def grid_to_coord(grid_x, grid_y):
    lat = 12.8 + (grid_x / GRID_SIZE) * (13.1 - 12.8)
    lon = 77.4 + (grid_y / GRID_SIZE) * (77.8 - 77.4)
    return (lat, lon)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html', drones=drones_data)

@app.route('/drones')
def drones():
    return jsonify(drones_data)

@app.route('/predict_path', methods=['POST'])
def predict_path_route():
    data = request.get_json()
    from_place = data.get('from')
    to_place = data.get('to')

    start_coords = get_coordinates_from_input(from_place)
    end_coords = get_coordinates_from_input(to_place)

    if not start_coords or not end_coords:
        return jsonify({"error": "One or both locations could not be found."}), 400

    start_grid = coord_to_grid(start_coords)
    end_grid = coord_to_grid(end_coords)

    # Safe ML-predicted, time-aware path with collision avoidance
    safe_path_with_time = get_safe_paths([start_grid], [end_grid], speeds=[1.0])[0]

    # Fix: unpack points correctly
    latlon_path = [grid_to_coord(pt[0], pt[1]) for (pt, time) in safe_path_with_time]
    latlon_path[0] = start_coords
    latlon_path[-1] = end_coords

    return jsonify({
        "path": latlon_path,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "start_name": from_place,
        "end_name": to_place
    })

@app.route('/predict_paths', methods=['POST'])
def predict_paths():
    data = request.get_json()
    drones = data.get('drones', [])

    starts, ends, speeds = [], [], []

    for drone in drones:
        from_place = drone['from']
        to_place = drone['to']
        speed = drone['speed']

        start_coords = get_coordinates_from_input(from_place)
        end_coords = get_coordinates_from_input(to_place)

        if not start_coords or not end_coords:
            return jsonify({"error": f"Could not find location for drone: {from_place} or {to_place}"}), 400

        starts.append(coord_to_grid(start_coords))
        ends.append(coord_to_grid(end_coords))
        speeds.append(speed)

    try:
        safe_paths = get_safe_paths(starts, ends, speeds)
        # Fix: unpack points correctly
        latlon_paths = [[grid_to_coord(pt[0], pt[1]) for (pt, time) in path] for path in safe_paths]
        return jsonify({ "paths": latlon_paths })
    except Exception as e:
        print("ML prediction error:", e)
        return jsonify({ "error": "ML prediction failed" }), 500

if __name__ == '__main__':
    app.run(debug=True)
