import os
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from .db import engine

GTD_PATH = os.getenv('GTD_PATH', '/data/gtd.csv')


def load_gtd_to_db():
    """Load GTD CSV into Postgres/PostGIS with robust handling and optional geom/index creation."""
    print('Loading GTD from', GTD_PATH)
    if not os.path.exists(GTD_PATH):
        print('GTD file not found at', GTD_PATH)
        return

    df = pd.read_csv(GTD_PATH, encoding='latin1', low_memory=False)

    # Detect and map common columns
    col_map = {}
    if 'eventid' in df.columns: col_map['eventid'] = 'eventid'
    if 'iyear' in df.columns: col_map['iyear'] = 'year'
    if 'imonth' in df.columns: col_map['imonth'] = 'month'
    if 'iday' in df.columns: col_map['iday'] = 'day'
    if 'latitude' in df.columns: col_map['latitude'] = 'latitude'
    if 'longitude' in df.columns: col_map['longitude'] = 'longitude'
    if 'country_txt' in df.columns: col_map['country_txt'] = 'country'
    if 'region_txt' in df.columns: col_map['region_txt'] = 'region'
    if 'attacktype1_txt' in df.columns: col_map['attacktype1_txt'] = 'attacktype'
    if 'nkill' in df.columns: col_map['nkill'] = 'nkill'
    if 'nwound' in df.columns: col_map['nwound'] = 'nwound'

    df = df.rename(columns=col_map)

    # Safe numeric conversion for dates
    df['year'] = pd.to_numeric(df.get('year', None), errors='coerce').fillna(0).astype(int)
    df['month'] = pd.to_numeric(df.get('month', 1), errors='coerce').fillna(1).astype(int)
    df['day'] = pd.to_numeric(df.get('day', 1), errors='coerce').fillna(1).astype(int)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']].assign(
        year=df['year'], month=df['month'], day=df['day']
    ), errors='coerce')

    # Ensure all required columns exist
    out_cols = ['eventid', 'year', 'month', 'day', 'date', 'latitude', 'longitude',
                'country', 'region', 'attacktype', 'nkill', 'nwound']
    for c in out_cols:
        if c not in df.columns:
            df[c] = None

    # Write to SQL (replace table)
    df[out_cols].to_sql('events', engine, if_exists='replace', index=False)
    print('Data loaded into events table')

    # Add geom column with PostGIS if available
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE events ADD COLUMN IF NOT EXISTS geom geometry(POINT,4326);"))
            conn.execute(text(
                "UPDATE events SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude),4326) "
                "WHERE geom IS NULL AND longitude IS NOT NULL AND latitude IS NOT NULL;"
            ))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_geom ON events USING GIST (geom);"))
        print('Geom column and index created successfully')
    except ProgrammingError:
        print('PostGIS not available — skipping geom/index creation')
    except Exception as e:
        print('Warning: could not create geom/index', e)


if __name__ == '__main__':
    load_gtd_to_db()
