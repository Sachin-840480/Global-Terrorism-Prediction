from fastapi import APIRouter, Query
from .db import engine
from .etl import load_gtd_to_db
from .hawkes_model import predict_risk

router = APIRouter()

@router.get('/events')
def get_events():
    # naive return - in production, add bbox and date filters
    import geopandas as gpd
    sql = 'SELECT id, date, latitude, longitude, nkill FROM events LIMIT 1000'
    gdf = gpd.read_postgis(sql, engine, geom_col='geom')
    return gdf.__geo_interface__

@router.get('/predict')
def get_predict(horizon_days: int = Query(90), agg: str = 'cell'):
    geojson = predict_risk(engine, horizon_days=horizon_days, agg=agg)
    return geojson

@router.get('/meta')
def meta():
    return {"model":"hawkes_stub_v0","default_horizon":90}