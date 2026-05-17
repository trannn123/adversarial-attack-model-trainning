import json
import matplotlib.pyplot as plt

# load dữ liệu
with open("evals_result.json", "r") as f:
    evals_result = json.load(f)

# lấy accuracy
train_acc = evals_result["train"]["accuracy"]
val_acc = evals_result["val"]["accuracy"]

# vẽ biểu đồ
plt.figure()

plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")

plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy over Training")
plt.legend()
plt.grid(True)
plt.savefig("plots/accuracy.png", dpi=300, bbox_inches="tight")
plt.show()