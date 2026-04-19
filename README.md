# AirLink — Autonomous Airspace Management System

## What is AirLink?

AirLink is an intelligent airspace management platform that enables **safe, efficient, and conflict-free drone navigation** in urban environments. It combines a trained neural network with real-time conflict avoidance to coordinate multiple UAVs simultaneously over Bengaluru's airspace.
**[https://airlink-fo5i.onrender.com](https://airlink-fo5i.onrender.com)**

---

## Features

- **Multi-drone coordination** — manage up to 8 UAVs simultaneously
- **ML-based path prediction** — MLP neural network predicts optimal flight waypoints
- **Automated collision avoidance** — unique altitude bands (50m, 80m, 110m...) guarantee 3D separation
- **Time-aware rerouting** — drones hover at contested waypoints to avoid space-time conflicts
- **Real-time map visualization** — animated drone icons on an interactive Leaflet map
- **Conflict reporting** — detailed report of any resolved conflicts and how they were handled
- **Auto altitude & speed assignment** — no manual inputs needed, the ML system handles it

---

## System Architecture

```
User Input (Origin + Destination)
        ↓
Flask Backend (app.py)
        ↓
Geocoding (Nominatim / OpenStreetMap)
        ↓
Grid Conversion (50×50 Bengaluru grid)
        ↓
MLP Model (model.pkl) → Waypoint Prediction
        ↓
Collision Resolver → Altitude Separation + Time-aware Hover
        ↓
Lat/Lon Conversion → Leaflet Map Visualization
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask |
| ML Model | scikit-learn MLPRegressor |
| Geocoding | geopy / Nominatim (OpenStreetMap) |
| Frontend | React 18 · Leaflet.js |
| Fonts | Rajdhani · Share Tech Mono |
| Deployment | Render |

---

## How It Works

1. Enter **Origin** and **Destination** for each drone in the ATC dashboard
2. Flask backend **geocodes** each location to a 50×50 grid over Bengaluru
3. The **MLP model** predicts 20 waypoints per drone along the grid
4. The **collision resolver** checks for space-time overlaps:
   - Each drone flies at a unique altitude band → guaranteed 3D separation
   - If two drones share the same grid zone at the same time → lower-priority drone hovers until safe
5. Safe paths are converted back to lat/lon and drawn on the **live map**
6. Animated drone icons fly along their paths in real time
7. The **ML Conflict Report** shows any resolved conflicts

---

## Future Enhancements

- [ ] 3D airspace visualization for better route planning
- [ ] AI-powered congestion prediction for high-traffic zones
- [ ] Blockchain-based authentication for secure drone communication
- [ ] Live telemetry feed from real drone hardware
- [ ] Pilot dashboards and drone registration modules
- [ ] Weather-aware path adjustment

---

## Team

Developed as part of an academic initiative:

| Name | GitHub |
|---|---|
| Navya B V | [@Navya-1008](https://github.com/Navya-1008) |
| Poorvi V Shetty | [@poorvi22-shetty](https://github.com/poorvi22-shetty) |
| Shravani Satish | — |
| Swaradaa Raghuram | [@Swaradaaraghuram](https://github.com/Swaradaaraghuram) |

