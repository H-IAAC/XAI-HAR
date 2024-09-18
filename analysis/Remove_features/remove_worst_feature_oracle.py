from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
import time

# import xAI techniques
from librep.xai.xai import (
    calc_shap_values,
    calc_shap_values_tree,
    shap_values_per_feature,
    calc_lime_values,
    lime_values_per_feature,
    calc_oracle_values,
    train_knn,
    train_rf,
    train_svm,
    train_dt,
    load_dataset,
)

import pickle
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import os
from multiprocessing import cpu_count

# Fix seed for reproducibility
Seed = 42
np.random.seed(Seed)


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

features_columns = [f"feature {i}" for i in range(24)]

classifiers = ["Random Forest", "SVM", "KNN", "Decision Tree"]
# reduce_on = ["all", "sensor", "axis"]
reduce_on = ["all"]
xai_techniques = ["ORACLE", "Random Forest", "SHAP", "LIME", "Tree"]
total_features = 24
path_to_save = Path("results/feature_importance_remove_worst_oracle")


############################################################################################################
# Function to run the experiment
############################################################################################################
def run_experiment(
    data_path: Path,
    dataset: str,
    reduce: str,
    classifier,
    removers: List[str],
    xai_technique: str,
    num_features: int=24,
    worst_feature: int=-1,
) -> None:
    """
    Function to run the experiment to remove the features using the xai technique.

    Parameters
    ----------
    data_path: Path
        The path to the data
    dataset: str
        The dataset name
    reduce: str
        The reduce type
    classifier: str
        The classifier name
    removers: List[str]
        The list of features to remove. For example: ["feature 1", "feature 2", "feature 3"]
    xai_technique: str
        The xai technique name
    num_features: int
        The number of features to remove
    worst_feature: int
        The worst feature index

    Returns
    -------
    None
    """

    # print(
    #     f"Dataset: {dataset}, reduce: {reduce}, classifier: {classifier}, xai technique: {xai_technique}, num_features: {num_features}"
    # )

    # print(f"Features to remove: {removers}")

    # Let's get the file name to verify if the file already exists or not
    file_name = f"{dataset}_{reduce}_{classifier}_{xai_technique}_{num_features}.pkl"
    # Let's check if the file already exists
    if Path(path_to_save / file_name).exists():
        result = pickle.load(open(path_to_save / file_name, "rb"))
        removers = result["features_to_remove"].copy()
        importances_df = result["feature importance"]
        worst_feature = result["Worst feature"]
        present_features = result["Present features"].copy()
        num_features = len(present_features) - 1

        if num_features == 0:
            return None
        else:
            run_experiment(
                data_path,
                dataset,
                reduce,
                classifier,
                removers,
                xai_technique,
                num_features,
                worst_feature,
            )
    else:
        # Let's calculate the number of features to remove that we take from the oracle technique
        # We need to remove the last features from the list to add the worst feature from the xai technique
        # n_removers = 24 - num_features - 1
        n_removers = 24 - num_features

        # Let's get the columns name's that we want to remove and the columns that we want to keep
        features_columns: List[str] = [f"feature {i}" for i in range(24)]
        present_features: List[str] = features_columns.copy()

        # Let's get the columns that we want to remove
        columns_to_remove: List[int] = []

        if n_removers > 1:
            for remover in removers[: n_removers - 1].copy():
                present_features.remove(remover)
                columns_to_remove += [features_columns.index(remover)]

            # We need verify if the worst feature is in the columns to remove, if it is the case, we need to get the second worst feature
            # Sometimes, the worst feature is the penultimate feature from oracle list, so we need to check, and if it is the case, we need to get the second worst feature
            # from the xai technique

            if worst_feature not in columns_to_remove:
                # We only need to add the worst feature to the columns to remove and remove this feature from the present features
                columns_to_remove += [worst_feature]
                present_features.remove(features_columns[worst_feature])

            else:
                # We need to get the second worst feature
                aux_num_features = num_features + 1
                aux_file_name = f"{dataset}_{reduce}_{classifier}_{xai_technique}_{aux_num_features}.pkl"
                aux_result = pickle.load(open(path_to_save / aux_file_name, "rb"))
                importances_df = aux_result["feature importance"]
                fis = importances_df.values[0]
                # Let's get the second worst feature
                idx = 1
                index_worst_feature = np.argsort(fis)[idx]
                column_worst_feature = importances_df.columns[index_worst_feature]
                other_worst_feature_idx = features_columns.index(column_worst_feature)

                while other_worst_feature_idx in columns_to_remove:
                    idx += 1
                    # Delete the variable column_worst_feature to get the next worst feature
                    del column_worst_feature
                    index_worst_feature = np.argsort(fis)[idx]
                    column_worst_feature = importances_df.columns[index_worst_feature]
                    other_worst_feature_idx = features_columns.index(
                        column_worst_feature
                    )

                # Add the index of the new column to remove
                columns_to_remove += [other_worst_feature_idx]
                # Remove the column name from the present features
                present_features.remove(column_worst_feature)
                # Update the worst feature
                worst_feature = other_worst_feature_idx

                # Update the worst feature in the aux file
                aux_result["Worst feature"] = worst_feature
                pickle.dump(aux_result, open(path_to_save / aux_file_name, "wb"))
                print(
                    f"We need to update the worst feature in the file {aux_file_name} for xai technique {xai_technique}, dataset {dataset}, reduce {reduce}, classifier {classifier}, num_features {num_features}"
                )
                print()
        elif n_removers == 1:
            columns_to_remove += [worst_feature]
            present_features.remove(features_columns[worst_feature])

        result = {
            "Dataset": dataset,
            "reduce on": reduce,
            "Classifier": classifier,
            "XAI technique": xai_technique,
            "features_to_remove": removers.copy(),
            "Present features": present_features.copy(),
            "Features removed": columns_to_remove.copy(),
            "object": None,
            "model": None,
            "feature importance": None,  # It is a dataframe with the feature importance
            "Accuracy": None,
            "Worst feature": None,
        }

        # Load the data
        train, test = load_dataset(dataset, reduce, normalization=None, path=data_path)
        # Let's remove the columns that we don't want
        if len(columns_to_remove) > 0:
            train.X = np.delete(train.X, columns_to_remove, axis=1)
            test.X = np.delete(test.X, columns_to_remove, axis=1)
        activities = np.unique(train.y)

        # Train the model
        model = (
            train_rf(train)
            if classifier == "Random Forest"
            else train_svm(train)
            if classifier == "SVM"
            else train_knn(train)
            if classifier == "KNN"
            else train_dt(train)
        )
        accuracy = model.score(test.X, test.y)

        result["Accuracy"] = accuracy
        result["model"] = model

        # Calculate the feature importance only if the number of features is greater than 1
        if num_features == 1:
            result["feature importance"] = None
            result["object"] = None
            result["features_to_remove"] = removers.copy()
            result["Worst feature"] = worst_feature
            # Save the result
            pickle.dump(result, open(path_to_save / file_name, "wb"))
            return None

        if xai_technique == "ORACLE":
            # Let's calculate the latent dimension, the size of the original data
            latent_dim = train.X.shape[1]

            feature_importance_by_oracle, accuracies = calc_oracle_values(
                classifier, dataset, reduce, latent_dim, columns_to_remove
            )
            importances_df = pd.DataFrame(
                [list(feature_importance_by_oracle)], columns=present_features
            )
            result["object"] = feature_importance_by_oracle
            # Let's get the worst feature, because it is part of sequence of features to remove
            worst_feature_idx = np.argmin(feature_importance_by_oracle)
            # Add the worst feature to the list of features to remove
            removers.append(present_features[worst_feature_idx])

            # Let's store all other accuracies when we remove the other features
            file_accuracy_name = f"accuracies.csv"
            df_accuracies = pd.DataFrame()
            df_accuracies["Accuracy"] = accuracies
            df_accuracies["Removed feature"] = importances_df.columns
            df_accuracies["Dataset"] = dataset
            df_accuracies["reduce on"] = reduce
            df_accuracies["Classifier"] = classifier
            df_accuracies["XAI"] = "Others"
            df_accuracies["Dimension"] = latent_dim - 1

            df_accuracies_old = pd.read_csv(path_to_save / file_accuracy_name) if Path(path_to_save / file_accuracy_name).exists() else None

            if df_accuracies_old is not None:
                df_accuracies = pd.concat([df_accuracies_old, df_accuracies])
                df_accuracies.reset_index(drop=True, inplace=True)
                df_accuracies.to_csv(path_to_save / file_accuracy_name, index=False)
            else:
                df_accuracies.to_csv(path_to_save / file_accuracy_name, index=False)

        elif xai_technique == "Random Forest":
            importances = model.feature_importances_
            result["object"] = importances
            importances_df = pd.DataFrame([list(importances)], columns=present_features)
        elif xai_technique == "Tree":
            importances = model.feature_importances_
            result["object"] = importances
            importances_df = pd.DataFrame([list(importances)], columns=present_features)

        elif xai_technique == "SHAP":
            shap_values = (
                calc_shap_values_tree(model, test)
                if classifier in ["Random Forest", "XGBoostClassifier"]
                else calc_shap_values(model, test)
            )
            result["object"] = shap_values
            feature_importance_by_shap = shap_values_per_feature(
                shap_values, activities, num_features
            )
            importances_df = feature_importance_by_shap.copy()

        elif xai_technique == "LIME":
            lime_values = calc_lime_values(model, test, train, standartized_codes)
            result["object"] = lime_values

            feature_importance_by_lime = lime_values_per_feature(
                lime_values,
                dataset,
                reduce,
                classifier,
                activities,
                standartized_codes,
                num_features,
            )
            importances_df = feature_importance_by_lime.copy()
            importances_df.drop(
                columns=["Classifier", "Dataset", "reduce on"], inplace=True
            )

        importances_df.columns = present_features
        column = importances_df.values.argmin()
        index_worst_feature = features_columns.index(importances_df.columns[column])
        worst_feature = index_worst_feature

        result["feature importance"] = importances_df.copy()
        result["features_to_remove"] = removers.copy()
        result["Worst feature"] = worst_feature

        # Save the result
        pickle.dump(result, open(path_to_save / file_name, "wb"))

        num_features -= 1
        run_experiment(
            data_path,
            dataset,
            reduce,
            classifier,
            removers,
            xai_technique,
            num_features,
            worst_feature,
        )


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
        required=False,
        default="/home/patrick/Documents/Repositories/hiaac-m4-experiments/analysis/XAI/reducer_experiments/results/execution/transformed_data",
    )

    args = parser.parse_args()
    data_path = Path(args.d)
    start_time = time.time()

    # Let's run the experiment
    os.makedirs(path_to_save, exist_ok=True) if not Path(
        path_to_save
    ).exists() else None

    experiments = product(xai_techniques, datasets, reduce_on, classifiers)
    valid_experiments = []
    for experiment in experiments:
        if experiment[0] == "Random Forest" and experiment[3] != "Random Forest":
            continue
        elif experiment[0] == "Tree" and experiment[3] != "Decision Tree":
            continue
        else:
            valid_experiments.append(experiment)

    for xai, dataset, reduce, classifier in tqdm(
        valid_experiments, total=len(valid_experiments)
    ):
        file_name = f"{dataset}_{reduce}_{classifier}_{'ORACLE'}_1.pkl"
        result = (
            pickle.load(open(path_to_save / file_name, "rb"))
            if Path(path_to_save / file_name).exists()
            else None
        )
        removers = result["features_to_remove"].copy() if result is not None else []
        worst_feature = -1
        run_experiment(
            data_path,
            dataset,
            reduce,
            classifier,
            removers,
            xai,
            num_features=24,
            worst_feature=worst_feature,
        )

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
