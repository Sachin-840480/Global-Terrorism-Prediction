import pandas as pd
from sqlalchemy import text
from datetime import datetime
import os
from .db import engine

GTD_PATH = os.getenv('GTD_PATH', '/data/gtd.csv')

def load_gtd_to_db():
    print('Loading GTD from', GTD_PATH)
    df = pd.read_csv(GTD_PATH, encoding='latin1', low_memory=False)
    # Minimal column mapping - adjust depending on GTD version
    df = df.rename(columns={
        'eventid':'eventid', 'iyear':'year','imonth':'month','iday':'day',
        'latitude':'latitude','longitude':'longitude','country_txt':'country',
        'region_txt':'region','attacktype1_txt':'attacktype','nkill':'nkill','nwound':'nwound'
    })
    df['date'] = pd.to_datetime(df[['year','month','day']].fillna(1))

    # write to SQL (naive approach)
    df[['eventid','year','month','day','date','latitude','longitude','country','region','attacktype','nkill','nwound']]
    df.to_sql('events', engine, if_exists='replace', index=False)
    # add geom column
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE events ADD COLUMN IF NOT EXISTS geom geometry(POINT,4326);"))
        conn.execute(text("UPDATE events SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude),4326) WHERE geom IS NULL;"))
    print('GTD loaded')

if __name__ == '__main__':
    load_gtd_to_db()