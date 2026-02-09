import json

path = r"data/ember2018/ember2018/train_features_0.jsonl"

with open(path, "r") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        print("Keys:", obj.keys())
        print("Label:", obj.get("label"))
        print("Feature dim:", len(obj["feature"]))
        break