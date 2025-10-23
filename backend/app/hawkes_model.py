"""
Expanded lightweight spatio-temporal "Hawkes-like" predictor.
This is NOT a production Hawkes implementation, but it captures
self-exciting behaviour by letting each past event contribute
an exponentially decaying risk in time and distance.

Features:
- Accepts bbox and aggregation mode (cell or country).
- Builds a regular lat/lon grid (configurable cell_size_deg).
- Computes per-cell risk = sum_over_events( severity_factor * temporal_decay * spatial_decay ).
- Caches results to /tmp for repeated horizon requests.
- Returns GeoJSON-like dict (FeatureCollection of polygons with "risk" property).
"""

from datetime import datetime, timedelta
import os
import math
import hashlib
import pickle

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# cache dir
CACHE_DIR = '/tmp/terrormap_cache'      # exponential temporal decay timescale
os.makedirs(CACHE_DIR, exist_ok=True)   # spatial decay lengthscale in km

# decay params (tunable)
TIME_DECAY_DAYS = 30.0
SPATIAL_SIGMA_KM = 200.0
EARTH_RADIUS_KM = 6371.0


def haversine_km(lon1, lat1, lon2, lat2):
    # compute great-circle distance
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def _cache_key(prefix, **kwargs):
    s = prefix + '|' + '|'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.sha1(s.encode()).hexdigest()


def _load_cache(key):
    path = os.path.join(CACHE_DIR, key + '.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_cache(key, obj):
    path = os.path.join(CACHE_DIR, key + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _make_grid(minx, miny, maxx, maxy, cell_size_deg=1.0):
    cols = int(math.ceil((maxx - minx) / cell_size_deg))
    rows = int(math.ceil((maxy - miny) / cell_size_deg))
    polys = []
    xs = []
    ys = []
    for i in range(cols):
        for j in range(rows):
            x1 = minx + i * cell_size_deg
            y1 = miny + j * cell_size_deg
            x2 = min(x1 + cell_size_deg, maxx)
            y2 = min(y1 + cell_size_deg, maxy)
            polys.append(box(x1, y1, x2, y2))
            xs.append((x1 + x2) / 2.0)
            ys.append((y1 + y2) / 2.0)
    gdf = gpd.GeoDataFrame({'geometry': polys, 'centroid_x': xs, 'centroid_y': ys})
    gdf.crs = 'EPSG:4326'
    return gdf


def predict_risk(db_engine, horizon_days=7, bbox=None, agg='cell', cell_size_deg=1.0, lookback_days=365):
    """
    Predict risk for the next `horizon_days` (used only for API labeling here).
    This function uses events in the last `lookback_days` to compute contributions.

    Returns GeoJSON-like FeatureCollection of polygons with property 'risk' (0..1 normalized).
    """
# bbox handling
    if bbox:
        try:
            if isinstance(bbox, str):
                minx, miny, maxx, maxy = map(float, bbox.split(','))
            else:
                minx, miny, maxx, maxy = bbox
        except Exception:
            minx, miny, maxx, maxy = -180.0, -90.0, 180.0, 90.0
    else:
        minx, miny, maxx, maxy = -180.0, -90.0, 180.0, 90.0

    cache_key = _cache_key('predict', bbox=f'{minx},{miny},{maxx},{maxy}', horizon=horizon_days, cell=cell_size_deg)
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    # read recent events
    since_date = (datetime.utcnow() - timedelta(days=lookback_days)).date()
    sql = f"SELECT id, eventid, date, latitude, longitude, nkill FROM events WHERE date >= '{since_date}' AND latitude IS NOT NULL AND longitude IS NOT NULL"
    try:
        df = pd.read_sql(sql, db_engine)
    except Exception:
        # fallback: empty dataframe
        df = pd.DataFrame(columns=['id','eventid','date','latitude','longitude','nkill'])

    if df.empty:
        # return empty grid with zero risk
        gdf_grid = _make_grid(minx, miny, maxx, maxy, cell_size_deg)
        gdf_grid['risk'] = 0.0
        out = gdf_grid[['geometry','risk']].__geo_interface__
        _save_cache(cache_key, out)
        return out

    # normalize/clean
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['nkill'] = pd.to_numeric(df['nkill'], errors='coerce').fillna(0.0)

    # compute event weight (severity factor)
    df['severity'] = (1.0 + df['nkill']) ** 0.5

    # compute time difference in days from now
    now = datetime.utcnow()
    df['tdays'] = (now - df['date']).dt.total_seconds() / (3600*24.0)

    # compute temporal decay
    df['t_decay'] = df['tdays'].clip(lower=0.0).apply(lambda t: math.exp(-t / TIME_DECAY_DAYS))

    # prepare grid
    gdf_grid = _make_grid(minx, miny, maxx, maxy, cell_size_deg)

    # compute risk contribution per grid centroid
    risks = []
    event_lons = df['longitude'].astype(float).values
    event_lats = df['latitude'].astype(float).values
    severity = df['severity'].values
    t_decay = df['t_decay'].values

    for cx, cy in zip(gdf_grid['centroid_x'].values, gdf_grid['centroid_y'].values):
        total = 0.0
        for elon, elat, sev, td in zip(event_lons, event_lats, severity, t_decay):
            try:
                d_km = haversine_km(cx, cy, elon, elat)
            except Exception:
                d_km = 10000.0
            s_decay = math.exp(-d_km / SPATIAL_SIGMA_KM)
            total += sev * td * s_decay
        risks.append(total)

    gdf_grid['raw_risk'] = risks

    # normalize raw_risk to 0..1
    maxr = gdf_grid['raw_risk'].max() if not gdf_grid['raw_risk'].isna().all() else 0.0
    if maxr > 0:
        gdf_grid['risk'] = gdf_grid['raw_risk'] / maxr
    else:
        gdf_grid['risk'] = 0.0

    # if country aggregation requested (simple zonal stats using naturalearth)
    if agg == 'country':
        try:
            countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            countries = countries.to_crs('EPSG:4326')
            joined = gpd.sjoin(gdf_grid, countries[['geometry','name']], how='left', predicate='intersects')
            zonal = joined.groupby('name')['risk'].mean().reset_index()
            features = []
            for _, row in zonal.iterrows():
                country_geom = countries.loc[countries['name'] == row['name'], 'geometry'].iloc[0]
                features.append({
                    'type':'Feature',
                    'geometry': gpd.GeoSeries([country_geom]).__geo_interface__['features'][0]['geometry'],
                    'properties': {'risk': float(row['risk']), 'name': row['name']}
                })
            out = {'type':'FeatureCollection', 'features': features}
            _save_cache(cache_key, out)
            return out
        except Exception:
            pass

    out = gdf_grid[['geometry','risk']].__geo_interface__
    _save_cache(cache_key, out)
    return out
