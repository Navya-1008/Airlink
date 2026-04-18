"""
ml_logic.py — AirLink ML path prediction and collision avoidance

Each drone is assigned a unique altitude band (50 m, 80 m, 110 m, …)
which guarantees 3-D separation. On top of that, a time-aware scheduler
makes lower-priority drones hover at contested waypoints to resolve any
2-D (top-down) space-time overlaps.
"""

import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor  # noqa: F401 (needed for pickle)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 50
STEPS     = 20

ALTITUDE_BASE = 50   # metres — lowest drone flies at 50 m
ALTITUDE_STEP = 30   # metres — each additional drone is 30 m higher

CONFLICT_TIME_WINDOW    = 2.0  # seconds — safety buffer around each waypoint
CONFLICT_SPATIAL_RADIUS = 1    # grid cells — neighbourhood radius for checks


# ---------------------------------------------------------------------------
# Path prediction
# ---------------------------------------------------------------------------

def predict_path(model, start: tuple, end: tuple) -> list:
    """
    Use the trained MLP to predict STEPS waypoints between start and end
    on the 50×50 grid.
    """
    path = []
    for i in range(1, STEPS + 1):
        input_data = [*start, *end, i]
        pred = model.predict([input_data])
        pred = tuple(int(np.clip(v, 0, GRID_SIZE - 1)) for v in pred[0])
        path.append(pred)
    return path


# ---------------------------------------------------------------------------
# Collision avoidance
# ---------------------------------------------------------------------------

def _cells_near(pt: tuple, radius: int = CONFLICT_SPATIAL_RADIUS) -> set:
    """Return all grid cells within `radius` of `pt`."""
    cells = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cells.add((pt[0] + dx, pt[1] + dy))
    return cells


def resolve_multi_drone_collisions(
    drone_paths: list,
    speeds: list,
    altitudes: list,
) -> tuple:
    """
    Time-aware collision avoidance with altitude separation.

    Altitude bands already guarantee 3-D separation.  This function
    additionally resolves 2-D (top-down) space-time overlaps by making
    lower-priority drones hover at contested waypoints.

    Returns:
        adjusted_paths  — list of [(grid_pt, timestamp), …] per drone
        conflicts       — list of conflict dicts (empty = no conflicts)
    """
    timeline: dict     = {}   # cell -> [(drone_idx, t_enter, t_exit)]
    adjusted_paths     = []
    conflicts_detected = []

    for idx, path in enumerate(drone_paths):
        speed        = max(speeds[idx], 0.01)
        duration     = 1.0 / speed   # time to cross one grid cell
        current_time = 0.0
        adjusted_path = []

        for pt in path:
            neighbours = _cells_near(pt)
            max_wait   = 0.0

            for cell in neighbours:
                for (other_idx, t_enter, t_exit) in timeline.get(cell, []):
                    if other_idx == idx:
                        continue

                    # Check for a time-window overlap
                    overlap = not (
                        current_time >= t_exit   + CONFLICT_TIME_WINDOW or
                        current_time + duration  <= t_enter - CONFLICT_TIME_WINDOW
                    )

                    if overlap:
                        wait_needed = (t_exit + CONFLICT_TIME_WINDOW) - current_time
                        if wait_needed > max_wait:
                            max_wait = wait_needed

                        conflicts_detected.append({
                            "drone_a":    other_idx,
                            "drone_b":    idx,
                            "cell":       list(pt),
                            "time":       round(current_time, 2),
                            "altitude_a": altitudes[other_idx],
                            "altitude_b": altitudes[idx],
                            "resolved_by": (
                                f"UAV-{str(idx + 1).zfill(3)} altitude separated "
                                f"({altitudes[idx]} m) and held "
                                f"{round(wait_needed, 1)} s"
                            ),
                        })

            current_time += max_wait   # hover to avoid conflict
            timeline.setdefault(pt, []).append(
                (idx, current_time, current_time + duration)
            )
            adjusted_path.append((pt, round(current_time, 2)))
            current_time += duration

        adjusted_paths.append(adjusted_path)

    return adjusted_paths, conflicts_detected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_safe_paths(starts: list, ends: list, speeds: list, altitudes: list) -> dict:
    """
    Main entry point called by Flask.

    Parameters
    ----------
    starts    : list of (grid_x, grid_y) tuples
    ends      : list of (grid_x, grid_y) tuples
    speeds    : list of float — grid-cells per second per drone
    altitudes : list of int  — altitude in metres per drone

    Returns
    -------
    dict with keys:
        paths     — list of [(grid_pt, timestamp), …]
        conflicts — list of resolved conflict dicts
        altitudes — altitudes used
    """
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            "model.pkl not found — run train_model.py first to generate it."
        )

    predicted_paths = [
        predict_path(model, starts[i], ends[i])
        for i in range(len(starts))
    ]

    safe_paths, conflicts = resolve_multi_drone_collisions(
        predicted_paths, speeds, altitudes
    )

    return {
        "paths":     safe_paths,
        "conflicts": conflicts,
        "altitudes": altitudes,
    }
