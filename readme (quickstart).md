# Terror Map — Quickstart

## Prereqs
- Docker & docker-compose
- GTD CSV downloaded and saved to `./data/gtd.csv` (see GTD registration steps)
- A Mapbox token (or set `USE_OSM=true` to use raster OSM tiles)

## Setup
1. Copy `.env.example` to `.env` and fill values (DATABASE_URL, MAPBOX_TOKEN, GTD_PATH if different).
2. Build & run:
   ```bash
   docker-compose up --build