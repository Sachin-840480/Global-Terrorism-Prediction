from fastapi import FastAPI
from .api import router
from .etl import load_gtd_to_db

app = FastAPI(title='TerrorMap API')
app.include_router(router, prefix='/api')

# On startup, try to load GTD if DB is empty
@app.on_event('startup')
def startup_event():
    try:
        load_gtd_to_db()
    except Exception as e:
        print('ETL load skipped or failed:', e)