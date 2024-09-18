# Let's create a script to create yaml files for execute the experiments
# We will use the same structure as the one used in the paper

import yaml
import os
from itertools import product
import tqdm

base_path = "../reducer_experiments/experiments"
if not os.path.exists(base_path):
    os.makedirs(base_path)

config_base = {
    "estimators": [
        {
            "algorithm": "RandomForest",
            "kwargs": {
                "n_estimators": 100,
            },
            "name": "random_forest-100",
            "num_runs": 10,
        },
        {
            "algorithm": "KNN",
            "kwargs": {
                "n_neighbors": 5,
            },
            "name": "knn-5",
            "num_runs": 1,
        },
        {
            "algorithm": "SVM",
            "kwargs": {
                "kernel": "rbf",
                "C": 1.0,
            },
            "name": "svm-rbf-C1.0",
            "num_runs": 1,
        },
    ],
    "extra": {
        "in_use_features": [
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        "reduce_on": "all",
        "scale_on": "train",
        "save_data": True,
    },
    "reducer": {
        "algorithm": "umap",
        "kwargs": {
            "n_components": 24,
            "random_state": 42,
        },
        "name": "umap-24",
        "use_y": False,
    },
    "reducer_dataset": [
        "kuhar.standartized_inter_balanced[train]",
        "kuhar.standartized_inter_balanced[validation]",
        "motionsense.standartized_inter_balanced[train]",
        "motionsense.standartized_inter_balanced[validation]",
        "wisdm.standartized_inter_balanced[train]",
        "wisdm.standartized_inter_balanced[validation]",
    ],
    "scaler": None,
    "transforms": [
        {
            "transform": "fft",
            "kwargs": {
                "centered": True,
            },
            "name": "fft",
            "windowed": {
                "fit_on": None,
                "transform_on": "window",
            },
        },
    ],
    "version": "1.0",
}

reduce_on = ["all", "axis", "sensor"]
datasets = [
    "kuhar",
    "motionsense",
    "uci",
    "wisdm",
    "realworld_thigh",
    "realworld_waist",
]


# Let's create the config files using cartesian product from reduce_on and datasets
for product in tqdm.tqdm(list(product(reduce_on, datasets))):
    dim = 24

    config = config_base.copy()
    reduce_on = product[0]
    dataset = product[1]

    config["train_dataset"] = [
        f"{dataset}.standartized_inter_balanced[train]",
        f"{dataset}.standartized_inter_balanced[validation]",
    ]
    config["test_dataset"] = [f"{dataset}.standartized_inter_balanced[test]"]

    if reduce_on == "sensor":
        dim = dim // 2
    elif reduce_on == "axis":
        dim = dim // 6

    config["extra"]["reduce_on"] = reduce_on
    config["reducer"]["kwargs"]["n_components"] = dim
    config["reducer"]["name"] = f"umap-{dim}"

    # Let's save the config file
    config_path = os.path.join(base_path, f"{dataset}-{reduce_on}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
