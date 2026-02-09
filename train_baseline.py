import joblib
from thrember import train_model, read_vectorized_features

DATA_DIR = "data"
MODEL_PATH = "models/baseline_lgbm.pkl"

print("[+] Training baseline model...")
model = train_model(DATA_DIR)

joblib.dump(model, MODEL_PATH)
print("[✓] Saved baseline model:", MODEL_PATH)
