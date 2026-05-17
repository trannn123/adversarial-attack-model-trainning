import lightgbm as lgb
import numpy as np
import os
from ember import PEFeatureExtractor

# config
ORIG_FOLDER = r"D:\malware_sample\sorel-dataset-PE-bin"
ADV_FOLDER  = "src/dataset/Gamma"
MODEL_PATH = "models/baseline_model.txt"

EXPECTED_FEATURES = 2568
THRESHOLD = 0.5

# load model
print("Loading model...")
model = lgb.Booster(model_file=MODEL_PATH)
extractor = PEFeatureExtractor()
print("Model loaded!\n")

# hàm đánh giá
def predict_file(path):
    try:
        with open(path, "rb") as f:
            bytez = f.read()
        features = extractor.feature_vector(bytez)
        features = np.array(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        current_dim = features.shape[1]
        if current_dim < EXPECTED_FEATURES:
            pad_width = EXPECTED_FEATURES - current_dim
            features = np.pad(
                features,
                ((0, 0), (0, pad_width)),
                mode="constant"
            )
        elif current_dim > EXPECTED_FEATURES:
            features = features[:, :EXPECTED_FEATURES]
        score = model.predict(features)[0]
        label = "malware" if score >= THRESHOLD else "benign"
        return float(score), label

    except Exception as e:
        print("Error processing:", path)
        print(e)
        return None, None

# loop
total = 0
success = 0
orig_scores = []
adv_scores = []

files = os.listdir(ORIG_FOLDER)
for file in files:
    if not file.endswith(".bin"):
        continue
    orig_path = os.path.join(ORIG_FOLDER, file)
    adv_path = os.path.join(ADV_FOLDER, "adv_" + file)

    if not os.path.isfile(adv_path):
        continue

    score_orig, _ = predict_file(orig_path)
    score_adv, _ = predict_file(adv_path)

    if score_orig is None or score_adv is None:
        continue

    total += 1

    orig_scores.append(score_orig)
    adv_scores.append(score_adv)

    if score_adv < THRESHOLD:
        success += 1
        result = "SUCCESS"
    else:
        result = "FAIL"
    print(f"{file}: {score_orig:.4f} -> {score_adv:.4f} | {result}")

# tổng hợp
if total > 0:
    avg_orig = float(np.mean(orig_scores))
    avg_adv  = float(np.mean(adv_scores))
    drop = avg_orig - avg_adv

    print("\n===== SUMMARY =====")
    print("Total samples:", total)
    print("Attack Success Rate (ASR):", round(success / total * 100, 2), "%")
    print("Avg malware score (original):", round(avg_orig, 4))
    print("Avg malware score (adversarial):", round(avg_adv, 4))
    print("Average drop:", round(drop, 4))
