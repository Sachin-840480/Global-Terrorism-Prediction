from fastapi import APIRouter, Query
from .db import engine
from .etl import load_gtd_to_db
from .hawkes_model import predict_risk

router = APIRouter()

@router.get('/events')
def get_events(bbox: str = Query(None), since: str = Query(None), until: str = Query(None)):
    """Return historical events as GeoJSON. Supports optional bbox filter and since/until."""
    import geopandas as gpd
    sql = 'SELECT id, eventid, date, latitude, longitude, nkill, geom FROM events'
    filters = []
    if since:
        filters.append(f"date >= '{since}'")
    if until:
        filters.append(f"date <= '{until}'")
    if filters:
        sql += ' WHERE ' + ' AND '.join(filters)
    sql += ' LIMIT 10000'
    gdf = gpd.read_postgis(sql, engine, geom_col='geom')
    if bbox:
        try:
            minx, miny, maxx, maxy = map(float, bbox.split(','))
            bbox_poly = gpd.GeoSeries([gpd.box(minx, miny, maxx, maxy)], crs='EPSG:4326')
            gdf = gdf[gdf.within(bbox_poly.iloc[0])]
        except Exception:
            pass
    return gdf.__geo_interface__

@router.get('/predict')
def get_predict(horizon_days: int = Query(90), bbox: str = Query(None), agg: str = 'cell', cell_size: float = 1.0):
    """Return predicted risk overlay as GeoJSON."""
    geojson = predict_risk(engine, horizon_days=horizon_days, bbox=bbox, agg=agg, cell_size_deg=cell_size)
    return geojson

@router.get('/meta')
def meta():
    return {"model":"hawkes_like_v1","default_horizon":90}
