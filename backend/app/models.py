from sqlalchemy import Column, Integer, Float, String, Date, Table
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry

Base = declarative_base()

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    eventid = Column(String, unique=True, index=True)
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    date = Column(Date)
    latitude = Column(Float)
    longitude = Column(Float)
    geom = Column(Geometry('POINT', srid=4326))
    country = Column(String)
    region = Column(String)
    attacktype = Column(String)
    nkill = Column(Integer)
    nwound = Column(Integer)