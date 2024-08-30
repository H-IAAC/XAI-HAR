from pathlib import Path
import os, shutil
import pandas as pd
from typing import Tuple, List, Dict, Any

from dataset_processor import (
    AddGravityColumn,
    AddSMV,
    ButterworthFilter,
    MedianFilter,
    CalcTimeDiffMean,
    Convert_G_to_Ms2,
    ResamplerPoly,
    Windowize,
    Peak_Windowize,
    AddStandardActivityCode,
    SplitGuaranteeingAllClassesPerSplit,
    BalanceToMinimumClass,
    BalanceToMinimumClassAndUser,
    LeaveOneSubjectOut,
    FilterByCommonRows,
    RenameColumns,
    Pipeline,
    GenerateFold,
)

# Set the seed for reproducibility
import numpy as np
import random

from utils import (
    read_kuhar,
    read_motionsense,
    read_wisdm,
    read_uci,
    read_realworld,
    read_recodGaitV1,
    read_recodGaitV2,
    read_GaitOpticalInertial,
    read_umafall,
    sanity_function,
    real_world_organize,
)

"""This module  used to generate the datasets. The datasets are generated in the following steps:    
    1. Read the raw dataset
    2. Preprocess the raw dataset
    3. Preprocess the standartized dataset
    4. Remove activities that are equal to -1
    5. Balance the dataset per activity
    6. Balance the dataset per user and activity
    7. Save the datasets
    8. Generate the views of the datasets
"""

random.seed(42)
np.random.seed(42)

# Variables used to map the activities from the RealWorld dataset to the standard activities
maping: List[int] = [4, 3, -1, -1, 5, 0, 1, 2]
tasks: List[str] = [
    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
standard_activity_code_realworld_map: Dict[str, int] = {
    activity: maping[tasks.index(activity)] for activity in tasks
}

har_datasets: List[str] = [
    "KuHar",
    "MotionSense",
    "UCI",
    "WISDM",
    "RealWorld",
]

authentications_datasets: List[str] = [
    "RecodGait_v1",
    "RecodGait_v2",
    "GaitOpticalInertial",
]

fall_datasets: List[str] = ["UMAFall"]
transitions_datasets: List[str] = ["HAPT", 'HAPT_different_transitions', 'HAPT_only_transitions']

column_group: Dict[str, str] = {
    "KuHar": ["user", "activity code", "csv"],
    "MotionSense": ["user", "activity code", "csv"],
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "activity code", "serial"],
    "RealWorld": ["user", "activity code", "position"],
    "RecodGait_v1": ["user", "index", "session"],
    "RecodGait_v2": ["user", "index", "session"],
    "GaitOpticalInertial": ["user", "session"],
    "UMAFall": ["file_name"],
    "HAPT": ["user", "activity code", "serial"],
    "HAPT_different_transitions": ["user", "activity code", "serial"],
    "HAPT_only_transitions": ["user", "activity code", "serial"],
}

'''
The standard activity code has the following mapping:
    0: Sitting
    1: Standing
    2: Walking
    3: Stair Up
    4: Stair Down
    5: Running
    6: Stars up and down
    7: 
    8: Laying
    9: Stand to sit
    10: Sit to stand
    11: Sit to lie
    12: Lie to sit
    13: Stand to lie
    14: Lie to stand
    15: ADL
    16: Fall
'''

standard_activity_code_map: Dict[str, Dict[Any, int]] = {
    "KuHar": {
        0: 1,   # Standing
        1: 0,   # Sitting
        2: -1,  # Talk-sit
        3: -1,  # Talk-stand
        4: -1,  # Stand-sit
        5: -1,  # Lay
        6: -1,  # Lay-stand
        7: -1,  # Pick
        8: -1,  # Jump
        9: -1,  # Push-up
        10: -1, # Sit-up
        11: 2,  # Walk
        12: -1, # Walk-backwards
        13: -1, # Walk-circle
        14: 5,  # Run
        15: 3,  # Stairs up
        16: 4,  # Stairs down
        17: -1, # Table-tennis
    },
    "KuHar_raw": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
    },
    "MotionSense": {
        0: 4,  
        1: 3, 
        2: 0, 
        3: 1, 
        4: 2, 
        5: 5
    },
    "MotionSense_raw": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "WISDM": {
        "A": 2,    # Walking
        "B": 5,    # Jogging
        "C": 6,    # Stairs
        "D": 0,    # Sitting
        "E": 1,    # Standing
        "F": -1,   # Typing
        "G": -1,   # Brushing teeth
        "H": -1,   # Eating soup
        "I": -1,   # Eating chips
        "J": -1,   # Eating pasta
        "K": -1,   # Drinking
        "L": -1,   # Eating sandwich
        "M": -1,   # Kicking
        "O": -1,   # Playing catch
        "P": -1,   # Dribbling
        "Q": -1,   # Writing
        "R": -1,   # Clapping
        "S": -1,   # Folding clothes
    },
    "WISDM_raw": {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "O": 13, "P": 14, "Q": 15, "R": 16, "S": 17,
    },
    "UCI": {
        1: 2,    # walk
        2: 3,    # stair up
        3: 4,    # stair down
        4: 0,    # sit
        5: 1,    # stand
        6: -1,   # Laying
        7: -1,   # stand to sit
        8: -1,   # sit to stand
        9: -1,   # sit to lie
        10: -1,  # lie to sit
        11: -1,  # stand to lie
        12: -1,  # lie to stand
    },
    "UCI_raw": {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
    },
    "RealWorld": standard_activity_code_realworld_map,
    "RealWorld_raw": { task: i for i, task in enumerate(tasks) },
    "RecodGait_v1": None,
    "RecodGait": None,
    "GaitOpticalInertial": None,
    "UMAFall": {
        # "ADL": 1,
        # "Fall": -1,
        1: 0, # ADL
        -1: 1, # Fall
    },
    "HAPT": {
        1: 2,    # walk
        2: 3,    # stair up
        3: 4,    # stair down
        4: 0,    # sit
        5: 1,    # stand
        6: 8,    # Laying
        7: 9,    # stand to sit
        8: 9,    # sit to stand
        9: 9,    # sit to lie
        10: 9,   # lie to sit
        11: 9,   # stand to lie
        12: 9,   # lie to stand
    },
}

columns_to_rename = {
    "KuHar": None,
    "MotionSense": {
        "userAcceleration.x": "accel-x",
        "userAcceleration.y": "accel-y",
        "userAcceleration.z": "accel-z",
        "rotationRate.x": "gyro-x",
        "rotationRate.y": "gyro-y",
        "rotationRate.z": "gyro-z",
    },
    "WISDM": None,
    "UCI": None,
    "RealWorld": None,
    "RecodGait_v1": None,
    "RecodGait_v2": None,
    "GaitOpticalInertial": {
        "acc x": "accel-x",
        "acc y": "accel-y",
        "acc z": "accel-z",
        "gyro x": "gyro-x",
        "gyro y": "gyro-y",
        "gyro z": "gyro-z",
    },
    "UMAFall": {
        "X-Axis": "accel-x",
        "Y-Axis": "accel-y",
        "Z-Axis": "accel-z",
    },
    "HAPT": None,
    "HAPT_different_transitions": None,
    "HAPT_only_transitions": None,
}

feature_columns: Dict[str, List[str]] = {
    "KuHar": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "MotionSense": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
        "attitude.roll",
        "attitude.pitch",
        "attitude.yaw",
        "gravity.x",
        "gravity.y",
        "gravity.z",
    ],
    "WISDM": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "UCI": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "RealWorld": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "RecodGait_v1": ["accel-x", "accel-y", "accel-z"],
    "RecodGait_v2": ["accel-x", "accel-y", "accel-z"],
    "GaitOpticalInertial": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ],
    "UMAFall": ["accel-x", "accel-y", "accel-z"],
    "HAPT": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
}

match_columns: Dict[str, List[str]] = {
    "KuHar": ["user", "serial", "window", "activity code"],
    "MotionSense": ["user", "serial", "window"],
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "serial", "window", "activity code"],
    "RealWorld": ["user", "window", "activity code", "position"],
    "RealWorld_thigh": ["user", "window", "activity code", "position"],
    "RealWorld_waist": ["user", "window", "activity code", "position"],
    "HAPT": ["user", "serial", "window", "activity code"],
}

pipelines: Dict[str, Dict[str, Pipeline]] = {
    "KuHar": {
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["KuHar"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["KuHar"],
                    up=2,
                    down=10,
                    groupby_column=column_group["KuHar"],
                ),
                Windowize(
                    features_to_select=feature_columns["KuHar"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["KuHar"],
                ),
                AddStandardActivityCode(standard_activity_code_map["KuHar"]),
            ]
        ),
    },
    "MotionSense": {
        "standartized_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["MotionSense"]),
                AddGravityColumn(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    gravity_columns=["gravity.x", "gravity.y", "gravity.z"],
                ),
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["MotionSense"],
                    up=2,
                    down=5,
                    groupby_column=column_group["MotionSense"],
                ),
                Windowize(
                    features_to_select=feature_columns["MotionSense"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["MotionSense"],
                ),
                AddStandardActivityCode(standard_activity_code_map["MotionSense"]),
            ]
        ),
    },
    "WISDM": {
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["WISDM"],
                    column_to_diff="timestamp-accel",
                    new_column_name="accel-timestamp-diff",
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=20,
                ),
                Windowize(
                    features_to_select=feature_columns["WISDM"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["WISDM"],
                ),
                AddStandardActivityCode(standard_activity_code_map["WISDM"]),
            ]
        ),
    },
    "UCI": {
        "standartized_dataset": Pipeline(
            [
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["UCI"],
                    up=2,
                    down=5,
                    groupby_column=column_group["UCI"],
                ),
                Windowize(
                    features_to_select=feature_columns["UCI"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["UCI"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UCI"]),
            ]
        ),
    },
    "RealWorld": {
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["RealWorld"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["RealWorld"],
                    up=2,
                    down=5,
                    groupby_column=column_group["RealWorld"],
                ),
                Windowize(
                    features_to_select=feature_columns["RealWorld"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["RealWorld"],
                ),
                AddStandardActivityCode(standard_activity_code_map["RealWorld"]),
            ]
        ),
    },
    "HAPT": {
        "standartized_dataset": Pipeline(
            [
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    Wn=0.3,
                    fs=50,
                    btype="high",
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["HAPT"],
                    up=2,
                    down=5,
                    groupby_column=column_group["HAPT"],
                ),
                Windowize(
                    features_to_select=feature_columns["HAPT"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["HAPT"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT"]),
            ]
        ),
    },
}

# Creating a list of functions to read the datasets
functions: Dict[str, callable] = {
    "KuHar": read_kuhar,
    "MotionSense": read_motionsense,
    "WISDM": read_wisdm,
    "UCI": read_uci,
    "RealWorld": read_realworld,
    "HAPT": read_uci,
}

dataset_path: Dict[str, str] = {
    "KuHar": "KuHar/1.Raw_time_domian_data",
    "MotionSense": "MotionSense/A_DeviceMotion_data",
    "WISDM": "WISDM/wisdm-dataset/raw/phone",
    "UCI": "UCI/RawData",
    "RealWorld": "RealWorld/realworld2016_dataset",
    "HAPT": "UCI/RawData",
}

# Preprocess the datasets

# Generate ../data/har folder
Path("../data/har").mkdir(parents=True, exist_ok=True) if not os.path.exists("../data/har") else None

view_path = {
    "standartized_dataset": "standartized_balanced",
}

# Path to save the datasets to har task
har_path = Path("../data/har")
har_path.mkdir(parents=True, exist_ok=True) if not os.path.exists("../data/har") else None

output_path_balanced: object = Path("../data/har/raw_balanced")
output_path_balanced_standartized: object = Path("../data/har/standartized_balanced")

output_path_balanced_standartized: object = Path("../data/har/standartized_balanced")

output_path_balanced_user: object = Path("../data/har/raw_balanced_user")
output_path_balanced_standartized_user: object = Path(
    "../data/har/standartized_balanced_user"
)

output_path_balanced_user: object = Path("../data/har/raw_balanced_user")
output_path_balanced_standartized_user: object = Path(
    "../data/har/standartized_balanced_user"
)

# Balncers and splitters used to har task
balancer_activity: object = BalanceToMinimumClass(class_column="standard activity code")
balancer_activity_and_user: object = BalanceToMinimumClassAndUser(
    class_column="standard activity code", filter_column="user", random_state=42
)

split_data: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.8,
    random_state=42,
)

split_data_train_val: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.9,
    random_state=42,
)

def balance_per_activity(
    dataset: str, dataframe: pd.DataFrame, output_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function balance the dataset per activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    train_df = balancer_activity(train_df)
    val_df = balancer_activity(val_df)
    test_df = balancer_activity(test_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

def split_per_user(dataset, dataframe, output_path):
    """The function balance the dataset per user and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

def balance_per_user_and_activity(dataset, dataframe, output_path):
    """The function balance the dataset per user and activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    train_df = balancer_activity_and_user(train_df)
    val_df = balancer_activity_and_user(val_df)
    test_df = balancer_activity_and_user(test_df)

    # new_df_balanced = balancer_activity_and_user(
    #     dataframe[dataframe["standard activity code"] != -1]
    # )

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

def generate_views(new_df, new_df_standartized, dataset):
    """This function generate the views of the dataset.

    Parameters
    ----------
    new_df : pd.DataFrame
        The raw dataset.
    new_df_standartized : pd.DataFrame
        The standartized dataset.
    dataset : str
        The dataset name.
    """

    random.seed(42)
    np.random.seed(42)

    # Preprocess and save the raw balanced dataset per user and activity
    balance_per_user_and_activity(
        dataset, new_df, output_path_balanced_user
    )

    # # Preprocess and save the raw balanced dataset per activity
    # balance_per_activity(
    #     dataset, new_df, output_path_balanced
    # )

    # # Preprocess and save the standartized balanced dataset per user and activity
    # balance_per_user_and_activity(
    #     dataset, new_df_standartized, output_path_balanced_standartized_user
    # )

    # Preprocess and save the standartized balanced dataset per activity
    balance_per_activity(
        dataset, new_df_standartized, output_path_balanced_standartized
    )

# ###########################################################################################################################
# This part of the code is used to generate the datasets views for the har task
# ###########################################################################################################################

# Creating the datasets views fot the har task
for dataset in har_datasets:

    print(f"Preprocessing the {dataset} dataset ...\n")
    os.mkdir(output_path_balanced) if not os.path.isdir(output_path_balanced) else None
    os.mkdir(output_path_balanced_standartized) if not os.path.isdir(output_path_balanced_standartized) else None
    os.mkdir(output_path_balanced_user) if not os.path.isdir(output_path_balanced_user) else None
    os.mkdir(output_path_balanced_standartized_user) if not os.path.isdir(output_path_balanced_standartized_user) else None

    reader = functions[dataset]

    # Verify if the file unbalanced.csv is already created

    # Read the raw dataset
    if dataset == "RealWorld":
        rw_thigh = "RealWorld_thigh"
        rw_waist = "RealWorld_waist"

        print("Organizing the RealWorld dataset ...\n")
        # Create a folder to save the organized dataset
        workspace = Path(
            "../data/original/RealWorld/realworld2016_dataset_organized"
        )

        os.mkdir(workspace) if not os.path.isdir(workspace) else None

        # Organize the dataset
        workspace, users = real_world_organize()
        path = workspace
        raw_dataset = reader(path, users)
        # Preprocess the raw dataset
        new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
        # Preprocess the standartized dataset
        new_df_standartized = pipelines[dataset]["standartized_dataset"](
            raw_dataset
        )

        # Remove activities that are equal to -1
        # new_df = new_df[new_df["standard activity code"] != -1]
        new_df_standartized = new_df_standartized[
            new_df_standartized["standard activity code"] != -1
        ]

        generate_views(new_df, new_df_standartized, dataset)
        
        positions = ["thigh", "waist"]
        for position in list(positions):
            print(f"Generating the views for the {dataset} dataset for the {position} position\n")
            new_df_filtered = new_df[new_df["position"] == position]
            new_df_standartized_filtered = new_df_standartized[
                new_df_standartized["position"] == position
            ]
            new_dataset = dataset + "_" + position

            generate_views(
                new_df_filtered, new_df_standartized_filtered, new_dataset
            )
            
    else:
        path = f"../data/original/{dataset_path[dataset]}"
        raw_dataset = reader(path)
        print(f"Total of activities in entire dataset: {raw_dataset['activity code'].nunique()}")

        # Preprocess the raw dataset
        new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
        print(f"Total of activities in entire dataset after the pipeline: {new_df['standard activity code'].nunique()}")

        # Preprocess the standartized dataset
        new_df_standartized = pipelines[dataset]["standartized_dataset"](
            raw_dataset
        )
        # Remove activities that are equal to -1
        # new_df = new_df[new_df["standard activity code"] != -1]
        new_df_standartized = new_df_standartized[
            new_df_standartized["standard activity code"] != -1
        ]
        generate_views(new_df, new_df_standartized, dataset)

# Remove the junk folder
workspace = Path("../data/processed")
if os.path.isdir(workspace):
    shutil.rmtree(workspace)

# Remove the realworld2016_dataset_organized folder
workspace = Path("../data/original/RealWorld/realworld2016_dataset_organized")
if os.path.isdir(workspace):
    shutil.rmtree(workspace)
