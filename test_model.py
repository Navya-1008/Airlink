"""
test_model.py — Quick sanity-check for model.pkl

Run after train_model.py to verify predictions look reasonable.

    python test_model.py
"""

import pickle
import numpy as np

STEPS = 20
START = (10.0, 10.0)
END   = (40.0, 40.0)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Testing path from {START} → {END}\n{'─' * 45}")
for i in range(1, STEPS + 1):
    input_data = [[*START, *END, i]]
    pred = model.predict(input_data)
    print(f"  Step {i:>2}: predicted waypoint = {pred[0].round(1)}")

print("\n✅  Test complete")
