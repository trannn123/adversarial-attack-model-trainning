import numpy as np
import lightgbm as lgb
import csv
import os
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

NUM_FEATURES = 2568

X_test = np.memmap("src/dataset/X_test.dat", dtype=np.float32, mode="r")
y_test = np.memmap("src/dataset/y_test.dat", dtype=np.int32, mode="r")
X_test = X_test.reshape(-1, NUM_FEATURES)

model = lgb.Booster(model_file="models/baseline_model.txt")

print("Evaluating...")

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("Accuracy:", acc)
print("Log Loss:", loss)
print("F1-score:", f1)
print("Confusion Matrix:")
print(cm)

# save metrics to file
file_exists = os.path.isfile("evaluation_results.csv")

with open("evaluation_results.csv", mode="a", newline="") as file:
    writer = csv.writer(file)

    # Ghi header nếu file chưa tồn tại
    if not file_exists:
        writer.writerow([
            "Model",
            "Accuracy",
            "LogLoss",
            "F1",
            "TN",
            "FP",
            "FN",
            "TP"
        ])

    writer.writerow([
        "baseline_new",
        acc,
        loss,
        f1,
        tn,
        fp,
        fn,
        tp
    ])
