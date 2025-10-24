import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

GRID_SIZE = 50
STEPS = 20

def predict_path(model, start, end):
    path = []
    for i in range(1, STEPS + 1):
        input_data = [*start, *end, i]
        print("Predict input:", input_data)  # Debug print
        pred = model.predict([input_data])
        print("Raw prediction:", pred)  # Debug print
        pred = tuple(map(int, pred[0]))
        path.append(pred)
    return path

def resolve_multi_drone_collisions(drone_paths, speeds):
    timeline = {}
    adjusted_paths = []

    for idx, path in enumerate(drone_paths):
        current_time = 0
        adjusted_path = []

        for pt in path:
            duration = 1 / speeds[idx]
            while any(start <= current_time < end for start, end in timeline.get(pt, [])):
                current_time += 0.5
            timeline.setdefault(pt, []).append((current_time, current_time + duration))
            adjusted_path.append((pt, round(current_time, 2)))
            current_time += duration

        adjusted_paths.append(adjusted_path)

    return adjusted_paths

def get_safe_paths(starts, ends, speeds):
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print("Error loading model.pkl:", e)
        raise e

    predicted_paths = []
    for i in range(len(starts)):
        try:
            path = predict_path(model, starts[i], ends[i])
            predicted_paths.append(path)
        except Exception as e:
            print(f"Prediction error for drone {i}: {e}")
            raise e

    safe_paths = resolve_multi_drone_collisions(predicted_paths, speeds)
    return safe_paths
