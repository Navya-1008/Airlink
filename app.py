import numpy as np
from flask import Flask, jsonify, request
from geopy.geocoders import Nominatim
from ml_logic import get_safe_paths, ALTITUDE_BASE, ALTITUDE_STEP

app = Flask(__name__)

GRID_SIZE = 50
STEPS = 20

# Default flight parameters — assigned automatically, no user input needed
DEFAULT_SPEED_MS   = 15.0    # metres per second (real-world)
GRID_CELL_METRES   = 600.0   # approx metres per grid cell across Bengaluru bbox
DEFAULT_SPEED_GRID = DEFAULT_SPEED_MS / GRID_CELL_METRES  # grid-cells per second


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_coordinates(place_name: str):
    """Geocode a place name to (lat, lon) using Nominatim."""
    geolocator = Nominatim(user_agent="airlink_atc")
    location = geolocator.geocode(place_name + ", Bengaluru, India")
    if location:
        return (location.latitude, location.longitude)
    return None


def coord_to_grid(coord):
    """Convert (lat, lon) to a 50x50 grid cell."""
    lat_norm = (coord[0] - 12.8) / (13.1 - 12.8)
    lon_norm = (coord[1] - 77.4) / (77.8 - 77.4)
    return (
        int(np.clip(lat_norm * GRID_SIZE, 0, GRID_SIZE - 1)),
        int(np.clip(lon_norm * GRID_SIZE, 0, GRID_SIZE - 1)),
    )


def grid_to_coord(grid_x, grid_y):
    """Convert a grid cell back to (lat, lon)."""
    lat = 12.8 + (grid_x / GRID_SIZE) * (13.1 - 12.8)
    lon = 77.4 + (grid_y / GRID_SIZE) * (77.8 - 77.4)
    return (lat, lon)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    # Served from static/ to bypass Jinja2 parsing React/JSX syntax
    return app.send_static_file('index.html')


@app.route('/predict_paths', methods=['POST'])
def predict_paths():
    data   = request.get_json()
    drones = data.get('drones', [])

    if not drones:
        return jsonify({"error": "No drones provided"}), 400

    starts, ends = [], []

    for i, drone in enumerate(drones):
        start_coords = get_coordinates(drone['from'])
        end_coords   = get_coordinates(drone['to'])

        if not start_coords or not end_coords:
            return jsonify({
                "error": (
                    f"Could not geocode locations for drone {i + 1}: "
                    f"'{drone['from']}' or '{drone['to']}'"
                )
            }), 400

        starts.append(coord_to_grid(start_coords))
        ends.append(coord_to_grid(end_coords))

    n = len(drones)

    # Auto-assign unique altitude bands — 50 m, 80 m, 110 m, …
    altitudes = [ALTITUDE_BASE + i * ALTITUDE_STEP for i in range(n)]

    # All drones fly at the same default speed
    speeds = [DEFAULT_SPEED_GRID] * n

    try:
        result = get_safe_paths(starts, ends, speeds, altitudes)
    except Exception as e:
        print("ML error:", e)
        return jsonify({"error": "ML prediction failed — ensure model.pkl exists"}), 500

    # Convert grid paths back to lat/lon for the frontend map
    latlon_paths = []
    for path in result["paths"]:
        latlon_paths.append([grid_to_coord(pt[0], pt[1]) for (pt, _) in path])

    # De-duplicate conflicts (one entry per drone pair)
    conflicts, seen = [], set()
    for c in result["conflicts"]:
        key = (c["drone_a"], c["drone_b"])
        if key not in seen:
            seen.add(key)
            conflicts.append({
                "drone_a":    c["drone_a"],
                "drone_b":    c["drone_b"],
                "altitude_a": c["altitude_a"],
                "altitude_b": c["altitude_b"],
                "resolved_by": c["resolved_by"],
            })

    return jsonify({
        "paths":              latlon_paths,
        "altitudes":          altitudes,
        "conflicts_detected": len(conflicts),
        "conflicts":          conflicts,
        "message": (
            "All paths safe — no conflicts detected." if not conflicts
            else f"{len(conflicts)} conflict(s) detected and resolved via "
                 "altitude separation and time-aware rerouting."
        ),
    })


if __name__ == '__main__':
    app.run(debug=True)
