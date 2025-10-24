import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Sample start and end points (grid coordinates, float)
start = (10.0, 10.0)  # example start point
end = (40.0, 40.0)    # example end point

STEPS = 20

def test_prediction():
    for i in range(1, STEPS + 1):
        input_data = [[start[0], start[1], end[0], end[1], i]]
        print(f"Input to model (step {i}): {input_data}")
        try:
            pred = model.predict(input_data)
            print(f"Model output (step {i}): {pred}")
        except Exception as e:
            print(f"Error during prediction at step {i}: {e}")

if __name__ == "__main__":
    test_prediction()
