# Global-Terrorism-Prediction

## Link:- https://global-terrorism-prediction.streamlit.app/

---

# Inside your .venv terminal for this project, run:

set STREAMLIT_CONFIG_DIR={path to file}\gtd\.streamlit

# Run The File

streamlit run gtd_dashboard_final.py


# Re-Train a New Model for experiments (NEW):

Just Go to `Root` and then run `python scripts/experiment_trainer.py`.

All the Model's details will be created in the experiments folder in the root.

# Re-Train a New Model for experiments (OLD):

1. Change the `MODEL_DIR = ROOT / "model"` to `MODEL_DIR = ROOT / "{experiments}"`.
2. Change or Add new Features in the `src/data/preprocessing.py`.
3. Change the model or Hyper-parameters in the `src/models/trainer.py.`
4. Come back to Root and run the following Command in the terminal:-

`python -c "from src.data.loader import load_data; from src.models.trainer import train_xgboost_cpu; df=load_data(); train_xgboost_cpu(df)"`
