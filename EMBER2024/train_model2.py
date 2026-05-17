import numpy as np
from lightgbm import LGBMClassifier

NUM_FEATURES = 2568

X_train = np.memmap("src/dataset/X_train.dat",
                    dtype=np.float32, mode="r")
y_train = np.memmap("src/dataset/y_train.dat",
                    dtype=np.int32, mode="r")

X_train = X_train.reshape(-1, NUM_FEATURES)

param_sets = [
    {
        "name": "baseline2",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 64
    },
    {
        "name": "robust",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 48
    },
    {
        "name": "strong",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 128
    }
]

for params in param_sets:
    print(f"Training {params['name']}...")

    model = LGBMClassifier(
        objective="binary",
        n_jobs=-1,
        **{k:v for k,v in params.items() if k!="name"}
    )

    model.fit(X_train, y_train)

    model.booster_.save_model(
        f"models/{params['name']}_model.txt"
    )

    print(f"{params['name']} done.")
