"""
train_model.py — Train and save the AirLink MLP path predictor.

Run this once to generate model.pkl before starting the Flask server.

    python train_model.py
"""

import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor

GRID_SIZE   = 50
STEPS       = 20
NUM_SAMPLES = 1000

X, y = [], []

for _ in range(NUM_SAMPLES):
    start = np.random.randint(0, GRID_SIZE, 2)
    end   = np.random.randint(0, GRID_SIZE, 2)

    for step in range(1, STEPS + 1):
        features = [*start, *end, step]
        X.append(features)

        ratio    = step / STEPS
        target_x = int(start[0] + ratio * (end[0] - start[0]))
        target_y = int(start[1] + ratio * (end[1] - start[1]))
        y.append([target_x, target_y])

X = np.array(X)
y = np.array(y)

print(f"Training on {len(X):,} samples …")
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅  Model trained and saved as model.pkl")
