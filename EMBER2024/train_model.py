import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import json

NUM_FEATURES = 2568

def lgb_accuracy(y_true, y_pred):
    y_pred_label = (y_pred > 0.5).astype(int)
    return 'accuracy', accuracy_score(y_true, y_pred_label), True

# load data
X = np.memmap("src/dataset/X_train.dat", dtype=np.float32, mode="r")
y = np.memmap("src/dataset/y_train.dat", dtype=np.int32, mode="r")
print(y[:10])
print(y.dtype)
X = X.reshape(-1, NUM_FEATURES)
# X = np.array(X)
# y = np.array(y)

# tách validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training...")

# lưu log
model = LGBMClassifier(
    objective="binary", # mặc định là binary cho LGBMClassifier
    n_estimators=500, # số lượng cây quyết định
    learning_rate=0.05, # tốc độ học
    num_leaves=128, # số lượng lá của cây trong mô hình cơ bản
    n_jobs=-1, # số lượng luồng song song được sử dụng cho quá trình huấn luyện, -1 là tất cả
    random_state=42 # giúp ổn định kết quả
)

# eval_set và metric
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=["train", "val"],
    eval_metric=["binary_logloss", "auc", lgb_accuracy]
)
evals_result = model.evals_result_
joblib.dump(model, "baseline_model.pkl")
model.booster_.save_model("baseline_model.txt")

print(evals_result.keys())
with open("evals_result.json", "w") as f:
    json.dump(evals_result, f, default=float)

# lưu validation để dùng lại
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

print("Training done. Model saved.")