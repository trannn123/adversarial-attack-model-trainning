import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

NUM_FEATURES = 2568

# load test data
X_test = np.memmap("src/dataset/X_test.dat", dtype=np.float32, mode="r")
y_test = np.memmap("src/dataset/y_test.dat", dtype=np.int32, mode="r")
X_test = X_test.reshape(-1, NUM_FEATURES)

# load model
model = lgb.Booster(model_file="baseline_model.txt")

print("Predicting...")

# predict
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

# tính % trên toàn bộ dataset
cm_percent = cm.astype('float') / cm.sum() * 100

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")

# ghi đè text: count + %
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        disp.text_[i, j].set_text(
            f"{cm[i, j]}\n({cm_percent[i, j]:.2f}%)"
        )

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("plots/confusion_matrix_percent_total.png", dpi=300, bbox_inches="tight")
plt.show()