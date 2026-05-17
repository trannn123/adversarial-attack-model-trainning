import json
import matplotlib.pyplot as plt

# load dữ liệu log
with open("evals_result.json", "r") as f:
    evals_result = json.load(f)

# loss
train_loss = evals_result["train"]["binary_logloss"]
val_loss = evals_result["val"]["binary_logloss"]

plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("plots/loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# auc
train_auc = evals_result["train"]["auc"]
val_auc = evals_result["val"]["auc"]

plt.figure()
plt.plot(train_auc, label="Train AUC")
plt.plot(val_auc, label="Validation AUC")
plt.xlabel("Iteration")
plt.ylabel("AUC")
plt.title("Training vs Validation AUC")
plt.legend()
plt.grid()
plt.savefig("plots/auc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()