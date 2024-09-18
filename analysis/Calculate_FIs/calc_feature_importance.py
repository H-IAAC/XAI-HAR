from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from librep.datasets.multimodal.operations import (
    DatasetFitter,
)

from librep.xai.xai import (
    load_dataset,
    train_rf,
    train_svm,
    train_knn,
    train_dt,
    calc_shap_values,
    calc_shap_values_tree,
    calc_lime_values,
    calc_oracle_values,
)

import argparse

# import xAI techniques
import shap
from lime import lime_tabular

import pickle
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time

# Fix seed for reproducibility
Seed = 42
np.random.seed(Seed)


"""This script calculates the feature importance using SHAP , LIME, Oracle and Gini index for each dataset, reduce on and classifier.
The datasets are: kuhar, motionsense, wisdm, uci, realworld_thigh and realworld_waist and the classifiers are: Random Forest, SVM, KNN and Decision Tree.
The reduce on are: all, sensor and axis.
For each dataset we make a set of preprocessing steps. The steps depend on the dataset and after the preprocessing we apply fft and apply umap trained with kuhar, motionsense, and 
wisdm datasets
to reduce the dimensionality to 24.
The reduce type all means that we apply umap to the dataset with 24 dimensions.
The reduce type sensor means that we apply two umaps, one to the accelerometer data and another to the gyroscope data. Each umap has 12 dimensions.
The reduce type axis means that we apply six umaps, one to the x axis, another to the y axis and another to the z axis of the accelerometer and gyroscope data. Each umap has 4 dimensions.
The steps of the preprocessing are:
    1. Add gravity acceleration to the accelerometer data
    2. Convert the accelerometer measurements from g to m/s^2
    3. Remove the gravity acceleration from the accelerometer data with a high pass filter Butterworth with cutoff frequency of 0.3 Hz
    4. Resample the data to 20 Hz
    5. Window the data with a window size of 3 seconds
    6. Add the column "standard activity code" to the dataset to standardize the activity codes

The following table shows the preprocessing steps for each dataset:
| Dataset           | Add gravity acceleration | Convert to m/s^2 | Remove gravity acceleration  | Resample | Window | Add column "standard activity code" |
|-------------------|--------------------------|------------------|------------------------------|----------|--------|-------------------------------------|
| kuhar             | No                       | No               | No                           | Yes      | Yes    | Yes                                 |
| motionsense       | Yes                      | Yes              | Yes                          | Yes      | Yes    | Yes                                 |
| wisdm             | No                       | No               | Yes                          | Yes      | Yes    | Yes                                 |
| uci               | No                       | No               | No                           | Yes      | Yes    | Yes                                 |
| realworld_thigh   | No                       | No               | No                           | Yes      | Yes    | Yes                                 |
| realworld_waist   | No                       | No               | No                           | Yes      | Yes    | Yes                                 |
"""

############################################################################################################
# Some variables to be used
############################################################################################################
standartized_codes = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

datasets = [
    "kuhar",
    "motionsense",
    "wisdm",
    "uci",
    "realworld_thigh",
    "realworld_waist",
]

reduce_on = ["all", "sensor", "axis"]

columns_features = ["Dataset", "reduce on", "Classifier"]
columns_features += [f"feature {i}" for i in range(24)]

columns_class = ["Dataset", "reduce on", "Classifier"] + list(
    standartized_codes.values()
)

classifiers = ["Random Forest", "SVM", "KNN", "Decision Tree"]

############################################################################################################
# Load the pickle files
############################################################################################################


def read_pickle_files(path: str = "results"):
    """Read the pickle files from the results folder and return a dictionary with the results"""

    columns_features = ["Dataset", "reduce on", "Classifier"]
    columns_features += [f"feature {i}" for i in range(24)]

    trustworthiness_columns = [
        "Dataset",  # Dataset used
        "reduce on",  # Reduce on used to reduce the dimensionality of the dataset
        "Classifier",  # Classifier used to predict the activity
        "sample_id",  # Index of the sample
        "standard activity",  # Standard activity code
        "Model prediction",  # Predicted activity code by the model
        "LIME prediction",  # Predicted activity code by LIME
        "Set",  # Test or Train
    ]

    # Read Oracle values from pickle file if it exists
    oracle_dir = Path(path + "/oracle_values.pickle")
    oracle_values_dict = (
        pickle.load(open(oracle_dir, "rb"))
        if oracle_dir.exists()
        else {
            dataset: {
                reduce: {classifier: None for classifier in classifiers}
                for reduce in reduce_on
            }
            for dataset in datasets
        }
    )

    # Read SHAP values from pickle file if it exists
    shap_dir = Path(path + "/shap_values.pickle")
    shap_values_dict = (
        pickle.load(open(shap_dir, "rb"))
        if shap_dir.exists()
        else {
            dataset: {
                reduce: {classifier: None for classifier in classifiers}
                for reduce in reduce_on
            }
            for dataset in datasets
        }
    )

    # Read LIME values from pickle file if it exists
    lime_dir = Path(path + "/lime_values.pickle")
    lime_values_dict = (
        pickle.load(open(lime_dir, "rb"))
        if lime_dir.exists()
        else {
            dataset: {
                reduce: {classifier: None for classifier in classifiers}
                for reduce in reduce_on
            }
            for dataset in datasets
        }
    )

    # Read feature importance values from pickle file if it exists
    rf_dir = Path(path + "/df_features_rf.pickle")
    df_features_rf = (
        pickle.load(open(rf_dir, "rb"))
        if rf_dir.exists()
        else pd.DataFrame(columns=columns_features)
    )

    dt_dir = Path(path + "/df_features_dt.pickle")
    df_features_dt = (
        pickle.load(open(dt_dir, "rb"))
        if dt_dir.exists()
        else pd.DataFrame(columns=columns_features)
    )

    # Read trustworthiness values from pickle file if it exists
    trustworthiness_dir = Path(path + "/df_trustworthines.pickle")
    df_trustworthines = (
        pickle.load(open(trustworthiness_dir, "rb"))
        if trustworthiness_dir.exists()
        else pd.DataFrame(columns=trustworthiness_columns)
    )

    return (
        oracle_values_dict,
        shap_values_dict,
        lime_values_dict,
        df_features_rf,
        df_features_dt,
        df_trustworthines,
    )


############################################################################################################
# Calculate the shap values and lime values for each dataset, reduce on and classifier
############################################################################################################
def calculate_fi(data_path):
    path_results = "results"
    if not Path(path_results).exists():
        Path(path_results).mkdir(parents=True, exist_ok=True)
    # Load the pickle files
    (
        oracle_values_dict,
        shap_values_dict,
        lime_values_dict,
        df_features_rf,
        df_features_dt,
        df_trustworthines,
    ) = read_pickle_files(path_results)

    # Calculate the feature importance for each dataset, reduce on and classifier
    for classifier, reduce, dataset in tqdm(
        list(product(classifiers, reduce_on, datasets))
    ):
        train, test = load_dataset(
            dataset, reduce, normalization=None, path=Path(data_path)
        )

        model = (
            train_rf(train)
            if classifier == "Random Forest"
            else train_svm(train)
            if classifier == "SVM"
            else train_knn(train)
            if classifier == "KNN"
            else train_dt(train)
        )

        # Let's verify if the xai thechniques were already calculated by the tuple (classifier, reduce, dataset)
        flag_oracle, flag_shap, flag_lime, flag_rf, flag_dt = (
            False,
            False,
            False,
            False,
            False,
        )

        for xai in ["oracle", "shap", "lime"]:
            doc = (
                oracle_values_dict
                if xai == "oracle"
                else shap_values_dict
                if xai == "shap"
                else lime_values_dict
            )
            if doc[dataset][reduce][classifier] is not None:
                flag_oracle = True if xai == "oracle" else flag_oracle
                flag_shap = True if xai == "shap" else flag_shap
                flag_lime = True if xai == "lime" else flag_lime

                if classifier == "Random Forest":
                    # Check if the feature importance was already calculated by the tuple (reduce, dataset)
                    if (
                        df_features_rf[
                            (df_features_rf["Dataset"] == dataset)
                            & (df_features_rf["reduce on"] == reduce)
                        ].shape[0]
                        > 0
                    ):
                        flag_rf = True

                if classifier == "Decision Tree":
                    # Check if the feature importance was already calculated by the tuple (reduce, dataset)
                    if (
                        df_features_dt[
                            (df_features_dt["Dataset"] == dataset)
                            & (df_features_dt["reduce on"] == reduce)
                        ].shape[0]
                        > 0
                    ):
                        flag_dt = True

        # If the xai techniques were not calculated, let's calculate them
        # Calculate the oracle values
        if not flag_oracle:
            oracle_values = calc_oracle_values(
                classifier, dataset, reduce, latent_dim=24, data_path=data_path
            )
            oracle_values_dict[dataset][reduce][classifier] = oracle_values

            # Save the oracle values in a pickle file
            pickle.dump(
                oracle_values_dict, open(path_results + "/oracle_values.pickle", "wb")
            )

        # Calculate the shap values
        if not flag_shap:
            shap_values = (
                calc_shap_values_tree(model, test)
                if classifier in ["Random Forest", "XGBoostClassifier", "Decision Tree"]
                else calc_shap_values(model, test)
            )
            shap_values_dict[dataset][reduce][classifier] = shap_values

            # Save the shap values in a pickle file
            pickle.dump(
                shap_values_dict, open(path_results + "/shap_values.pickle", "wb")
            )

        # Calculate the lime values
        if not flag_lime:
            lime_values = calc_lime_values(model, test, train, standartized_codes)
            lime_values_dict[dataset][reduce][classifier] = lime_values

            # Save the lime values in a pickle file
            pickle.dump(
                lime_values_dict, open(path_results + "/lime_values.pickle", "wb")
            )

        # Calculate the feature importance for Random Forest or Decision Tree
        if classifier == "Random Forest" and not flag_rf:
            importances = model.feature_importances_
            df = pd.DataFrame(
                [[dataset, reduce, classifier] + list(importances)],
                columns=columns_features,
            )

            df_features_rf = pd.concat([df_features_rf, df], ignore_index=True)

            # Save the feature importance dataframe in a pickle file
            pickle.dump(
                df_features_rf, open(path_results + "/df_features_rf.pickle", "wb")
            )

        if classifier == "Decision Tree" and not flag_dt:
            importances = model.feature_importances_
            df = pd.DataFrame(
                [[dataset, reduce, classifier] + list(importances)],
                columns=columns_features,
            )

            df_features_dt = pd.concat([df_features_dt, df], ignore_index=True)

            # Save the feature importance dataframe in a pickle file
            pickle.dump(
                df_features_dt, open(path_results + "/df_features_dt.pickle", "wb")
            )
            
        # Calculate the trustworthiness level for each sample
        for i, sample in enumerate(test.X):
            # Check if the trustworthiness was already calculated by the tuple (reduce, dataset, classifier)
            if (
                df_trustworthines[
                    (df_trustworthines["Dataset"] == dataset)
                    & (df_trustworthines["reduce on"] == reduce)
                    & (df_trustworthines["Classifier"] == classifier)
                ].shape[0]
                == 0
            ):
                sample_id = i
                standard_activity = test.y[i]
                model_predict = model.predict(sample.reshape(1, -1))[0]
                lime_prediction = lime_values[i]["LIME prediction"]

                df = pd.DataFrame(
                    [
                        [
                            dataset,
                            reduce,
                            classifier,
                            sample_id,
                            standard_activity,
                            model_predict,
                            lime_prediction,
                            "Test",
                        ]
                    ],
                    columns=df_trustworthines.columns,
                )

                df_trustworthines = pd.concat(
                    [df_trustworthines, df], ignore_index=True
                )

                # Save the trustworthiness dataframe in a pickle file
                pickle.dump(
                    df_trustworthines,
                    open(path_results + "/df_trustworthines.pickle", "wb"),
                )

    # Fininshing the execution of the function
    print("Finished the execution of the function")


############################################################################################################
# Main function
############################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the feature importance using SHAP and LIME",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "-data-path",
        type=str,
        action="store",
        help="Dataset to be used",
        required=True,
        default="../reducer_experiments/results/execution/transformed_data",
    )

    args = parser.parse_args()
    print(args)

    start_time = time.time()

    # Calculate the shap values and lime values for each dataset, reduce on and classifier
    calculate_fi(args.d)

    final_time = time.time() - start_time
    # Print the final in the folow example:
    # The experiment took 1 week, 5 days, 1 hour, 2 minutes and 3 seconds
    weeks = int(final_time / (60 * 60 * 24 * 7))
    days = int((final_time % (60 * 60 * 24 * 7)) / (60 * 60 * 24))
    hours = int((final_time % (60 * 60 * 24)) / (60 * 60))
    minutes = int((final_time % (60 * 60)) / 60)
    seconds = int(final_time % 60)
    print(
        f"The experiment took {weeks} week, {days} days, {hours} hours, {minutes} minutes and {seconds} seconds"
    )
