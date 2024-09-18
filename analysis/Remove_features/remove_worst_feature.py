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
from copy import copy

from sklearn.metrics import accuracy_score, f1_score

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

classifiers = ["Random Forest", "Decision Tree", "SVM", "KNN"]
reduce_on = ["all", "sensor", "axis"]
# reduce_on = ["all"]
xai_techniques = ["ORACLE", "Random Forest", "SHAP", "LIME", "Tree"]
total_features = 24
path_to_save = Path("results/feature_importance_remove_worst")


############################################################################################################
# Function to run the experiment
############################################################################################################
def run_experiment(
    data_path: Path,
    dataset: str,
    reduce: str,
    classifier,
    xai_technique: str,
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
    xai_technique: str
        The xai technique name

    Returns
    -------
    None
    """

    num_features = total_features

    # Load the dataset
    train, test = load_dataset(
        dataset_name=dataset,
        reduce_on=reduce,
        path=data_path,
    )

    activities = np.unique(train.y)

    model = (
        train_rf(train)
        if classifier == "Random Forest"
        else train_svm(train)
        if classifier == "SVM"
        else train_knn(train)
        if classifier == "KNN"
        else train_dt(train)
    )

    features_columns = [f"feature {i}" for i in range(24)]

    # Calculate the feature importance
    if xai_technique == "ORACLE":
        latent_dim = copy(num_features)
        feature_importance, _ = calc_oracle_values(
            classifier, dataset, reduce, latent_dim,
        )
        importances_df = pd.DataFrame(
                [list(feature_importance)], columns=features_columns,
            )                    

    elif xai_technique == "Random Forest":
        feature_importance = model.feature_importances_
        importances_df = pd.DataFrame(
                [list(feature_importance)], columns=features_columns,
            )
        
    elif xai_technique == "Tree":
        feature_importance = model.feature_importances_
        importances_df = pd.DataFrame(
                [list(feature_importance)], columns=features_columns,
            )
        
    elif xai_technique == "SHAP":
        shap_values = (
            calc_shap_values_tree(model, test)
            if classifier in ["Random Forest", "Decision Tree"]
            else calc_shap_values(model, test)
        )
        feature_importance = shap_values_per_feature(
            shap_values, activities, num_features
        )
        importances_df = feature_importance.copy()

    elif xai_technique == "LIME":
        lime_values = calc_lime_values(model, test, train, standartized_codes)
        feature_importance = lime_values_per_feature(
                lime_values,
                dataset,
                reduce,
                classifier,
                activities,
                standartized_codes,
                num_features,
            )
        
        importances_df = feature_importance[features_columns].copy()
    
    # Order the columns by features importance
    features_ordered = np.argsort(importances_df.values)[0]
    columns_to_remove = []

    while(num_features > 0):
        file_name = f"{dataset}_{reduce}_{classifier}_{xai_technique}_accuracy.csv"
        if Path(path_to_save / file_name).exists():
            accuracy_results = pd.read_csv(path_to_save / file_name)
        
        else:
            accuracy_results = pd.DataFrame(columns=["Classifier", "Dataset", "Reduce", "XAI", "Accuracy", "F1-Score", "Dimension", "Feature Removed"])

        if accuracy_results[accuracy_results[['Classifier', 'Dataset', 'Reduce', 'XAI', 'Dimension']].eq([classifier, dataset, reduce, xai_technique, num_features]).all(axis=1)].shape[0] > 0:
            num_features -= 1
            continue
        else:
                
            train_copy = copy(train)
            test_copy = copy(test)

            # Remove the worst feature
            columns_to_remove = features_ordered[:(total_features-num_features)]

            train_copy.X = np.delete(train_copy.X, columns_to_remove, axis=1)
            test_copy.X = np.delete(test_copy.X, columns_to_remove, axis=1)

            # Train the model
            model = (
                train_rf(train_copy)
                if classifier == "Random Forest"
                else train_svm(train_copy)
                if classifier == "SVM"
                else train_knn(train_copy)
                if classifier == "KNN"
                else train_dt(train_copy)
            )


            # Calculate the accuracy
            accuracy = accuracy_score(test_copy.y, model.predict(test_copy.X))
            f1 = f1_score(test_copy.y, model.predict(test_copy.X), average="macro")   

            # Store results
            results = {
                "Classifier": classifier,
                "Dataset": dataset,
                "Reduce": reduce,
                "XAI": xai_technique,
                "Accuracy": accuracy,
                "F1-Score": f1,
                "Dimension": num_features,
                "Feature Removed": features_ordered[total_features - num_features],
            }
            df = pd.DataFrame([results])
            accuracy_results = pd.concat([accuracy_results, df], ignore_index=True)
            accuracy_results.reset_index(drop=True, inplace=True)
            accuracy_results.to_csv(path_to_save / file_name, index=False)

            num_features -= 1

            print(f"Num features: {num_features}", "XAI: ", xai_technique, "Classifier: ", classifier, "Dataset: ", dataset, "Reduce: ", reduce)

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

    experiments = product(datasets, reduce_on, classifiers, xai_techniques)
    valid_experiments = []
    for experiment in experiments:
        if experiment[3] == "Random Forest" and experiment[2] != "Random Forest":
            continue
        elif experiment[3] == "Tree" and experiment[2] != "Decision Tree":
            continue
        else:
            valid_experiments.append(experiment)

    for dataset, reduce, classifier, xai in tqdm(
        valid_experiments, total=len(valid_experiments)
    ):
        
        run_experiment(
            data_path,
            dataset,
            reduce,
            classifier,
            xai,
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
