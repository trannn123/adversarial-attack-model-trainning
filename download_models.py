import os
import thrember

MODEL_DIR = r"./models"

os.makedirs(MODEL_DIR, exist_ok=True)

thrember.download_models(MODEL_DIR)
