# Terrorism Prediction Map

# Terror Map — Quickstart

This repository is a PoC full-stack application for visualising historical terrorism incidents and producing short-term risk predictions using a Hawkes-like spatio-temporal heuristic. The default prediction horizon is 90 days.

## Prerequisites
- Docker & docker-compose (recommended)
- GTD CSV downloaded and saved to `./data/gtd.csv` (see https://www.start.umd.edu/gtd/)
- Mapbox token (optional) or use OSM tiles by setting `USE_OSM=true` in `.env`

## Run (Docker Compose)
1. Copy `.env.example` to `.env` and fill values (DATABASE_URL, MAPBOX_TOKEN, GTD_PATH if different).
2. Place `gtd.csv` in the `./data/` folder.
3. Start services:
   ```bash
   docker-compose up --build

Predict and visualize terrorism activity using GTD dataset.