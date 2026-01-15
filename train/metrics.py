import sys
import json
import pprint
metrics_file = sys.argv[1]

with open(metrics_file, "r") as f:
    data = json.loads(f.readlines()[0])
    metrics = list(data["metrics"].keys())

metrics_dict = {}
for m in metrics:
    metrics_dict[m] = 0
metrics_dict["training_time"] = 0
metrics_dict["batch_size"] = 0

with open(metrics_file,"r") as f:
    folds = len(f.readlines())
    f.seek(0)
    for line in f.readlines():
        data = json.loads(line)
        for d in metrics:
            metrics_dict[d] += data["metrics"][d]
        metrics_dict["training_time"] += (data["training_time"]/3600)
    for m in metrics:
        metrics_dict[m] = metrics_dict[m]/folds

print("AVERAGE VALUES -",folds,"FOLDS")
pprint.pprint(metrics_dict)

metrics_dict["file"] = metrics_file
metrics_dict["folds"] = folds
metrics_dict["batch_size"] = metrics_dict["batch_size"]

with open("aggregated-results.jsonl", "a+") as f:
    f.write(json.dumps(metrics_dict)+"\n")