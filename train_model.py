import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

GRID_SIZE = 50
STEPS = 20
num_samples = 1000

X = []
y = []

# Generate synthetic training data
for _ in range(num_samples):
    start = np.random.randint(0, GRID_SIZE, 2)
    end = np.random.randint(0, GRID_SIZE, 2)

    for step in range(1, STEPS + 1):
        features = [*start, *end, step]
        X.append(features)

        ratio = step / STEPS
        target_x = int(start[0] + ratio * (end[0] - start[0]))
        target_y = int(start[1] + ratio * (end[1] - start[1]))
        y.append([target_x, target_y])

X = np.array(X)
y = np.array(y)

# Train the model
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
model.fit(X, y)

# Save to model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
