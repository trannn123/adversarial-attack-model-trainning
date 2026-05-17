import numpy as np
import lightgbm as lgb
from ember import PEFeatureExtractor

# load model
model = lgb.Booster(model_file="models/baseline_model.txt")
print("Model loaded")

# Feature extractor (2568 features)
extractor = PEFeatureExtractor(feature_version=2)

# load malware
malware_path = r"D:/malware_sample/sorel-dataset-PE-bin/0003840cfc486058c7e6de0d7e14d3683982e37a61e0d4a883409a22f980ef13.bin"

with open(malware_path, "rb") as f:
    bytez = f.read()

raw = extractor.raw_features(bytez)
x_mal = extractor.process_raw_features(raw)
if len(x_mal) < 2568:
    x_mal = np.pad(x_mal, (0, 2568 - len(x_mal)))
print("Feature size:", len(x_mal))

# feature importance
importance = model.feature_importance()
top_k = 50 # tìm 50 feature quan trọng
important_idx = np.argsort(importance)[-top_k:]
print("Top important features:", top_k)

# original prediction
prob_original = model.predict(x_mal.reshape(1, -1))[0]
print("\nOriginal malware probability:", prob_original)

# feature-space attack
x_adv = x_mal.copy()

steps = 40
alpha = 0.5
decay = 0.1
prob = model.predict(x_adv.reshape(1, -1))[0]

print("\nAttack progress\n")

for step in range(steps):
    improved = False
    for i in important_idx:
        original = x_adv[i]
        # thử tăng
        for delta in [alpha, alpha / 2, alpha / 4]:

            x_adv[i] = original + delta # thử tăng trước vì tăng giống benign
            prob_new = model.predict(x_adv.reshape(1, -1))[0]
            print(f"[TRY +] Feature {i} | {original:.4f} -> {x_adv[i]:.4f} | prob: {prob:.6f} -> {prob_new:.6f}")
            if prob_new < prob:
                print(f"[ACCEPT +] Feature {i} improved!")
                prob = prob_new
                improved = True
                break

        # thử giảm
        else:
            x_adv[i] = original * (1 - decay) #vì có feature càng nhỏ càng tốt vd entropy
            prob_new = model.predict(x_adv.reshape(1, -1))[0]
            print(f"[TRY -] Feature {i} | {original:.4f} -> {x_adv[i]:.4f} | prob: {prob:.6f} -> {prob_new:.6f}")
            if prob_new < prob:
                print(f"[ACCEPT -] Feature {i} improved!")
                prob = prob_new
                improved = True
                continue

            # nếu không cải thiện thì rollback
            else:
                x_adv[i] = original

    print("Step", step+1, "-> malware probability:", prob)

    if not improved:
        print("[STUCK] Applying random perturbation")

        for _ in range(5):  # thử 5 lần random
            i = np.random.choice(important_idx)

            original = x_adv[i]
            noise = np.random.uniform(-0.5, 0.5) # thêm nhiễu ngẫu nhiên

            x_adv[i] = original + noise
            prob_new = model.predict(x_adv.reshape(1, -1))[0]
            print(f"[RANDOM] Feature {i} | noise {noise:.4f} | prob: {prob:.6f} -> {prob_new:.6f}")
            if prob_new < prob:
                prob = prob_new
                print("[ESCAPE] Found better direction!")
                improved = True
                break
            else:
                x_adv[i] = original

        if not improved:
            print("[STOP] Cannot escape local minimum")
            break

    if prob < 0.5:
        print("[SUCCESS] Attack succeeded!")
        break

print("\nFinal probability:", prob)
# lưu lại dạng feature đã bị tác động để load dùng để ánh xạ qua file thật
np.save("x_adv.npy", x_adv)
print("Saved x_adv!")