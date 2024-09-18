# Remove deprecated warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Python imports
import argparse
from collections import defaultdict
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports
import coloredlogs
import ray
import tqdm
import yaml
from config import *
from dacite import from_dict
import pickle

# Librep imports
from librep.config.type_definitions import PathLike
from librep.datasets.har.loaders import PandasMultiModalLoader
from librep.datasets.multimodal import (
    ArrayMultiModalDataset,
    MultiModalDataset,
    TransformMultiModalDataset,
    WindowedTransform,
)
from librep.base.evaluators import SupervisedEvaluator
from librep.metrics.report import ClassificationReport
from librep.utils.workflow import MultiRunWorkflow, SimpleTrainEvalWorkflowMultiModal
from ray.util.multiprocessing import Pool
from utils import catchtime, load_yaml, get_sys_info, multimodal_multi_merge

"""This module is used to execute the experiments based on configuration files,
written in YAML. The configuration files are writen in YAML and the valid keys 
are defined in the ExecutionConfig in config.py file. Each YAML configuration 
file is loaded using the dacite library, which converts it into a ExecutionConfig 
dataclass. The ExecutionConfig is used to configure the experiment and then run it.

The experiment is run using the ray library, which allows to run the experiment
in parallel. The number of parallel processes is, by default, defined by the 
number of CPUs available on the machine.

The valid values for the configuration file are defined in the config.py file.
This includes:
- Estimator names and classes (in the estimator_cls variable)
- Scaler names and classes (in the scaler_cls variable)
- Reducer names and classes (in the reducer_cls variable)
- Transform names and classes (in the transform_cls variable)
- Dataset names and paths (in the datasets variable)

The code is divided into four main parts:
1. The main function (at the end), which is used to parse the command line arguments 
    and  call the `run_single_thread` (for sequential execution) or `run_ray` 
    (for parallel execution).
2. The `run_single_thread` and the `run_ray` functions calls the `run_wrapper` function.
    The `run_single_thread` function calls the `run_wrapper` function sequentially.
    The `run_ray` function calls the `run_wrapper` function in parallel, for each 
    configuration file, using ray.
3. The `run_wrapper` function, put a exception handling before calling the `run_experiment`
    function. 
4. The `run_experiment` function is the main function of the module, that actually runs
    the experiment. It is responsible for loading the configuration file, loading the 
    datasets, running the experiment and saving the results. 
    This experiment is controled by a `ExecutionConfig` object, which is passed to the
    function as a parameter. This object is created from the YAML configuration file.
    The `run_experiment` function calls the utilitary functions defined in this module.
"""

# Uncomment to remove warnings
# warnings.filterwarnings("always")


def load_datasets(
    dataset_locations: Dict[str, PathLike],
    datasets_to_load: List[str],
    label_columns: str = "standard activity code",
    features: List[str] = (
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ),
    dataset_structure: DatasetStructure = None,
) -> ArrayMultiModalDataset:
    """Utilitary function to load the datasets.
    It load the datasets from specified in the `datasets_to_load` parameter.
    The datasets specified are concatenated into a single ArrayMultiModalDataset.
    This dataset is then returned.

    Parameters
    ----------
    datasets_to_load : List[str]
        A list of datasets to load. Each dataset is specified as a string in the
        following format: "dataset_name.dataset_view[split]". The dataset name is the name
        of the dataset as specified in the `datasets` variable in the config.py
        file. The split is the split of the dataset to load. It can be either
        "train", "validation" or "test".
    label_columns : str, optional
        The name of column that have the label, by default "standard activity code"
    features : List[str], optional
        The features to load, from datasets
        by default ( "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z", )

    Returns
    -------
    ArrayMultiModalDataset
        An ArrayMultiModalDataset with the loaded datasets (concatenated).

    Examples
    --------
    >>> load_datasets(
    ...     dataset_locations={
    ...         "kuhar.standartized_balanced": "data/kuhar",
    ...         "motionsense.standartized_balanced": "data/motionsense",
    ...     },
    ...     datasets_to_load=[
    ...         "kuhar.standartized_balanced[train]",
    ...         "kuhar.standartized_balanced[validation]",
    ...         "motionsense.standartized_balanced[train]",
    ...         "motionsense.standartized_balanced[validation]",
    ...     ],
    ... )
    """
    # Transform it to a Path object
    dset_names = set()
    # if dataset_structure is None:
    #     dataset_structure = DatasetStructure()

    if dataset_structure is None:
        dataset_structure = DatasetStructure()
    data_type = dataset_structure.type
    user_id = dataset_structure.user_id
    fold = dataset_structure.fold
    load_train = dataset_structure.load_train
    load_validation = dataset_structure.load_validation
    load_test = dataset_structure.load_test
    # Remove the split from the dataset name
    # dset_names will contain the name of the datasets to load
    # it is used to index the datasets variable in the config.py file
    for dset in datasets_to_load:
        name = dset.split("[")[0]
        dset_names.add(name)

    # Load the datasets
    multimodal_datasets = dict()
    for name in dset_names:
        # Define dataset path. Join the root_dir with the path of the dataset
        path = dataset_locations[name]
        if data_type == 'traditional':
            pass

        elif data_type == 'k-fold':
            if user_id:
                new_path = path / f"{user_id}"
                path = new_path if new_path.exists() else path
            new_path = path / f"{fold}"
            path = new_path if new_path.exists() else path
            
        elif data_type == 'user':
            new_path = path / f"{user_id}"
            path = new_path if new_path.exists() else path

        # Load the dataset
        loader = PandasMultiModalLoader(root_dir=path)
        train, validation, test = loader.load(
            load_train=load_train,
            load_validation=load_validation,
            load_test=load_test,
            as_multimodal=True,
            as_array=True,
            features=features,
            label=label_columns,
        )
        
        # Store the multiple MultimodalDataset in a dictionary
        multimodal_datasets[name] = {
            "train": train,
            "validation": validation,
            "test": test,
        }

    # Concatenate the datasets

    # Pick the name and the split of the first dataset to load
    name = datasets_to_load[0].split("[")[0]
    split = datasets_to_load[0].split("[")[1].split("]")[0]
    final_dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])

    # Pick the name and the split of the other datasets to load and
    # Concatenate the other datasets
    for dset in datasets_to_load[1:]:
        name = dset.split("[")[0]
        split = dset.split("[")[1].split("]")[0]
        dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])
        final_dset = ArrayMultiModalDataset.concatenate(final_dset, dset)

    return final_dset


# Non-parametric transform
def do_transform(
    datasets: Dict[str, MultiModalDataset],
    transform_configs: List[TransformConfig],
    keep_suffixes: bool = True,
) -> Dict[str, MultiModalDataset]:
    """Utilitary function to apply a list of transforms to datasets

    Parameters
    ----------
    datasets : Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective dataset.
    transform_configs : List[TransformConfig]
        List of the transforms to apply. Each transform it will be instantiated
        based on the TransformConfig object and each one will be applied to the
        datasets.
    keep_suffixes : bool, optional
        Keep the window name suffixes, by default True

    Returns
    -------
    Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective transformed dataset.
    """
    new_datasets = dict()
    # Loop over the datasets
    for dset_name, dset in datasets.items():
        transforms = []
        new_names = []

        # Loop over the transforms and instantiate them
        for transform_config in transform_configs:
            # Get the transform class and kwargs and instantiate the transform
            kwargs = transform_config.kwargs or {}
            the_transform = transforms_cls[transform_config.transform](**kwargs)
            # If the transform is windowed, instantiate the WindowedTransform
            # with the defined fit_on and transform_on.
            if transform_config.windowed:
                the_transform = WindowedTransform(
                    transform=the_transform,
                    fit_on=transform_config.windowed.fit_on,
                    transform_on=transform_config.windowed.transform_on,
                )
            # Else instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            else:
                the_transform = WindowedTransform(
                    transform=the_transform,
                    fit_on=None,
                    transform_on="window",
                )
            # Create the list of transforms to apply to the dataset
            transforms.append(the_transform)
            if keep_suffixes:
                new_names.append(transform_config.name)

        new_name_prefix = ".".join(new_names)
        if new_name_prefix:
            new_name_prefix += "."

        # Instantiate the TransformMultiModalDataset with the list of transforms
        transformer = TransformMultiModalDataset(
            transforms=transforms, new_window_name_prefix=new_name_prefix
        )
        # Apply the transforms to the dataset
        dset = transformer(dset)
        # Append the transformed dataset to the list of new datasets
        new_datasets[dset_name] = dset

    # Return the list of transformed datasets
    return new_datasets


# Parametric transform
def do_reduce(
    datasets: Dict[str, MultiModalDataset],
    reducer_config: ReducerConfig,
    reducer_dataset_name="reducer_dataset",
    reducer_validation_dataset_name="reducer_validation_dataset",
    reduce_on: str = "all",
    suffix: str = "reduced",
    use_y: bool = False,
    apply_only_in: List[str] = None,
    sensor_names: List[str] = ("accel", "gyro"),
) -> Dict[str, MultiModalDataset]:
    """Utilitary function to perform dimensionality reduce to a list of
    datasets. The first dataset will be used to fit the reducer. And the
    reducer will be applied to the remaining datasets.

    Parameters
    ----------
    datasets : Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective dataset.
    reducer_config : ReducerConfig
        The reducer configuration, used to instantiate the reducer.
    reducer_dataset_name : str, optional
        The name of the dataset to use to fit the reducer, by default "reducer_dataset"
    reducer_validation_dataset_name : str, optional
        The name of the dataset to use to fit the reducer, by default "reducer_validation_dataset"
    reduce_on : str, optional
        How reduce will perform, by default "all".
        It can have the following values:
        - "all": the reducer will be applied to the whole dataset.
        - "sensor": the reducer will be applied to each sensor, and then,
            the datasets will be concatenated.
        - "axis": the reducer will be applied to each axis of each sensor,
            and then, the datasets will be concatenated.
    suffix : str, optional
        The new suffix to be appended to the window name, by default "reduced."
    use_y: bool, optional
        If True, the reducer will be trained with X and y (supervised). If False, the reducer
        will be trained with X only. By default False.
    apply_only_in: List[str], optional
        List of datasets to apply the reducer. If None, the reducer will be applied to all
        excluding the reducer_dataset_name and reducer_validation_dataset_name. By default None.
    sensor_names : List[str], optional
        The sensor names, by default ("accel", "gyro")

    Returns
    -------
    Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective transformed dataset.
    Raises
    ------
    ValueError
        - If the number of datasets is less than 2.
        - If the reduce_on value is invalid.

    NotImplementedError
        If the reduce_on is not implemented yet.
    """
    #  ----- Sanity check -----

    # The reducer_dataset_name must be in the datasets
    if reducer_dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{reducer_dataset_name}' not found. "
            + f"Maybe you forgot to load it in your configuration file? "
            + f"Check if any 'reducer_dataset' is defined in your configuration file."
        )

    if apply_only_in is not None:
        for dset_name in apply_only_in:
            if dset_name not in datasets:
                raise ValueError(
                    f"Dataset '{dset_name}' not found. "
                    + f"Maybe you forgot to load it in your configuration file?"
                )
    else:
        apply_only_in = list(datasets.keys())

    # Remove reducer_dataset_name and reducer_validation_dataset_name from apply_only_in
    apply_only_in = [
        dset_name
        for dset_name in apply_only_in
        if dset_name not in (reducer_dataset_name, reducer_validation_dataset_name)
    ]

    # Get the reducer kwargs
    kwargs = reducer_config.kwargs or {}

    # Output datasets
    new_datasets = {k: v for k, v in datasets.items()}

    # If reduce on is "all", fit the reducer on the first dataset and
    # apply the reducer to the remaining datasets
    if reduce_on == "all":
        # Get the reducer class and instantiate it using the kwargs
        reducer = reducers_cls[reducer_config.algorithm](**kwargs)
        # Fit the reducer on the reducer_dataset_name
        fit_dsets = {
            "X": datasets[reducer_dataset_name][:][0],
        }
        # If use_y is True, train the reducer with X and y
        if use_y:
            fit_dsets["y"] = datasets[reducer_dataset_name][:][1]
        # If the reducer_validation_dataset_name is in the datasets, use it
        if reducer_validation_dataset_name in datasets:
            fit_dsets["X_val"] = datasets[reducer_validation_dataset_name][:][0]
        # If the reducer_validation_dataset_name is in the datasets and use_y is True,
        # use it
        if reducer_validation_dataset_name in datasets and use_y:
            fit_dsets["y_val"] = datasets[reducer_validation_dataset_name][:][1]

        # Fit the reducer the datasets specified in fit_dsets
        reducer.fit(**fit_dsets)

        # Instantiate the WindowedTransform with fit_on=None and
        # transform_on="all", i.e. the transform will be applied to
        # whole dataset.
        transform = WindowedTransform(
            transform=reducer,
            fit_on=None,
            transform_on="all",
        )
        # Instantiate the TransformMultiModalDataset with the list of transforms
        # and the new suffix
        transformer = TransformMultiModalDataset(
            transforms=[transform], new_window_name_prefix=suffix
        )
        # Apply the transform to the remaining datasets
        new_datasets.update(
            {dset_name: transformer(datasets[dset_name]) for dset_name in apply_only_in}
        )

    # If reduce on is "sensor" or "axis", fit the reducer on each sensor
    # and if reduce on is "axis", fit the reducer on each axis of each sensor
    elif reduce_on == "sensor" or reduce_on == "axis":
        if reduce_on == "axis":
            window_names = datasets["reducer_dataset"].window_names
        else:
            window_names = [
                [w for w in datasets["reducer_dataset"].window_names if s in w]
                for s in sensor_names
            ]
            window_names = [w for w in window_names if w]

        window_datasets = defaultdict(list)

        # Loop over the windows (accel, gyro, for "sensor"; (accel-x, accel-y, accel-z, gyro-x, gyro-y, gyro-z, for "axis")
        for i, wname in enumerate(window_names):
            # Get the reducer class and instantiate it using the kwargs
            reducer = reducers_cls[reducer_config.algorithm](**kwargs)
            # Fit the reducer on the first dataset
            reducer_window = datasets["reducer_dataset"].windows(wname)
            # Fit the reducer on the reducer_dataset_name
            fit_dsets = {
                "X": datasets[reducer_dataset_name].windows(wname)[:][0],
            }
            # If use_y is True, train the reducer with X and y
            if use_y:
                fit_dsets["y"] = datasets[reducer_dataset_name].windows(wname)[:][1]
            # If the reducer_validation_dataset_name is in the datasets, use it
            if reducer_validation_dataset_name in datasets:
                fit_dsets["X_val"] = datasets[reducer_validation_dataset_name].windows(
                    wname
                )[:][0]
            # If the reducer_validation_dataset_name is in the datasets and use_y is True,
            # use it
            if reducer_validation_dataset_name in datasets and use_y:
                fit_dsets["y_val"] = datasets[reducer_validation_dataset_name].windows(
                    wname
                )[:][1]

            # Fit the reducer the datasets specified in fit_dsets
            reducer.fit(**fit_dsets)

            # Instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            transform = WindowedTransform(
                transform=reducer,
                fit_on=None,
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            # and the new suffix
            transformer = TransformMultiModalDataset(
                transforms=[transform], new_window_name_prefix=f"{suffix}-{i}"
            )

            # Apply the transform on the same window of each dataset
            # in apply_only_in
            for dataset_name in apply_only_in:
                dset_window = datasets[dataset_name].windows(wname)
                dset_window = transformer(dset_window)
                window_datasets[dataset_name].append(dset_window)

        # Merge dataset windows
        new_datasets.update(
            {
                dset_name: multimodal_multi_merge(window_datasets[dset_name])
                for dset_name in apply_only_in
            }
        )

    else:
        raise ValueError(
            "Invalid reduce_on value. Must be one of: 'all', 'axis', 'sensor"
        )

    # if reducer_dataset_name in datasets:
    #     new_datasets[reducer_dataset_name] = datasets[reducer_dataset_name]
    # if reducer_validation_dataset_name in datasets:
    #     new_datasets[reducer_validation_dataset_name] = datasets[
    #         reducer_validation_dataset_name
    #     ]
    return new_datasets


# Scaling transform
def do_scaling(
    datasets: Dict[str, MultiModalDataset],
    scaler_config: ScalerConfig,
    scale_on: str = "self",
    suffix: str = "scaled.",
    apply_only_in: List[str] = ("train_dataset", "validation_dataset", "test_dataset"),
    train_dataset_name="train_dataset",
) -> Dict[str, MultiModalDataset]:
    """Utilitary function to perform scaling to a list of datasets.
    If scale_on is "self", the scaling will be fit and transformed applied
    to each dataset. If scale_on is "train", the scaling will be fit to the
    first dataset and then, the scaling will be applied to all the datasets.
    (including the first one, that is used to fit the model).

    Parameters
    ----------
    datasets : List[MultiModalDataset]
        The list of datasets to scale. The first dataset will be used to fit
        the scaler if scale_on is "train".
    scaler_config : ScalerConfig
        The scaler configuration, used to instantiate the scaler.
    scale_on : str, optional
        How scaler will perform, by default "self".
        It can have the following values:
        - "self": the scaler will be fit and transformed applied to each dataset.
        - "train": the scaler will be fit to the first dataset and then, the
            scaling will be applied to all the datasets.
    suffix : str, optional
        The new suffix to be appended to the window name, by default "scaled."
    apply_only_in: List[str], optional
        List of datasets to apply the scaler.
    train_dataset_name: str, optional
        If scale_on is "train", this parameter is used to specify the name of
        the dataset to use to fit the scaler.

    Returns
    -------
    Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective scaled dataset.

    Raises
    ------
    ValueError
        - If the scale_on value is invalid.
    """
    #
    kwargs = scaler_config.kwargs or {}
    if scale_on == "self":
        # Loop over the datasets
        for dataset_name in apply_only_in:
            # Get the dataset
            dataset = datasets[dataset_name]
            # Get the scaler class and instantiate it using the kwargs
            transform = scaler_cls[scaler_config.algorithm](**kwargs)
            # Fit the scaler usinf the whole dataset and (i.e., fit_on="all")
            # and then, apply the transform to the whole dataset (i.e.,
            # transform_on="all")
            windowed_transform = WindowedTransform(
                transform=transform,
                fit_on="all",
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            transformer = TransformMultiModalDataset(
                transforms=[windowed_transform], new_window_name_prefix=suffix
            )
            # Apply the transform to the dataset
            dataset = transformer(dataset)
            # Append the dataset to the list of new datasets
            datasets[dataset_name] = dataset

        return datasets

    elif scale_on == "train":
        # Get the scaler class and instantiate it using the kwargs
        transform = scaler_cls[scaler_config.algorithm](**kwargs)
        # Fit the scaler on the first dataset
        transform.fit(datasets[train_dataset_name][:][0])
        for dataset_name in apply_only_in:
            dataset = datasets[dataset_name]
            # Instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            windowed_transform = WindowedTransform(
                transform=transform,
                fit_on=None,
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            transformer = TransformMultiModalDataset(
                transforms=[windowed_transform], new_window_name_prefix=suffix
            )
            # Apply the transform to the dataset
            dataset = transformer(dataset)
            # Append the dataset to the list of new datasets
            datasets[dataset_name] = dataset
        return datasets
    else:
        raise ValueError(f"scale_on: {scale_on} is not valid")


def do_classification(
    datasets: Dict[str, MultiModalDataset],
    estimator_config: EstimatorConfig,
    reporter: SupervisedEvaluator,
    train_dataset_name="train_dataset",
    validation_dataset_name="validation_dataset",
    test_dataset_name="test_dataset",
) -> dict:
    """Utilitary function to perform classification to a list of datasets.

    Parameters
    ----------
    datasets : Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective dataset.
    estimator_config : EstimatorConfig
        The estimator configuration, used to instantiate the estimator.
    reporter : SupervisedEvaluator
        The reporter object, used to evaluate the model.

    Returns
    -------
    dict
        Dictionary with the results of the experiment.
    """
    results = dict()

    # Get the estimator class and instantiate it using the kwargs
    estimator = estimator_cls[estimator_config.algorithm](
        **(estimator_config.kwargs or {})
    )
    # Instantiate the SimpleTrainEvalWorkflowMultiModal
    workflow = SimpleTrainEvalWorkflowMultiModal(
        estimator=estimator,
        do_fit=True,
        evaluator=reporter,
    )
    # Instantiate the MultiRunWorkflow
    runner = MultiRunWorkflow(workflow=workflow, num_runs=estimator_config.num_runs)
    # Run the workflow
    with catchtime() as classification_time:
        results["results"] = runner(
            datasets[train_dataset_name],
            datasets[validation_dataset_name]
            if validation_dataset_name in datasets
            else None,
            datasets[test_dataset_name],
        )

    results["classification_time"] = float(classification_time)
    results["estimator"] = asdict(estimator_config)

    return results


# Function that runs the experiment
def run_experiment(
    dataset_locations: Dict[str, PathLike],
    experiment_output_file: PathLike,
    config_to_execute: ExecutionConfig,
) -> dict:
    """This function is the wrapper that runs the experiment.
    The experiment is defined by the config_to_execute parameter,
    which controls the experiment execution.

    This code runs the following steps (in order):
    1. Load the datasets
    2. Perform the non-parametric transformations, if any, using `do_transform`
        function. The transforms are specified by `config_to_execute.transforms`
        which is a list of `TransformConfig` objects.
    3. Perform the parametric transformations, if any, using `do_reduce` function.
        The reducer algorithm and parameters are specified by
        `config_to_execute.reducers` which `ReducerConfig` object.
    4. Perform the scaling, if any, using `do_scaling` function. The scaler
        algorithm and parameters are specified by `config_to_execute.scaler`
        which is a `ScalerConfig` object.
    5. Perform the training and evaluation of the model.
    6. Save the results to a file.

    Parameters
    ----------
    dataset_locations :  Dict[str, PathLike],
        Dictionary with dataset locations. Key is the dataset name and value
        is the path to the dataset.
    experiment_output_file : PathLike
        Path to the file where the results will be saved.
    config_to_execute : ExecutionConfig
        The configuration of the experiment to be executed.

    Returns
    -------
    dict
        Dictionary with the results of the experiment.

    Raises
    ------
    ValueError
        If the reducer is specified but the reducer_dataset is not specified.
    """
    experiment_output_file = Path(experiment_output_file)

    if config_version != config_to_execute.version:
        raise ValueError(
            f"Config version ({config_to_execute.version}) "
            f"does not match the current version ({config_version})"
        )

    # Useful variables
    additional_info = dict()
    start_time = time.time()

    # Dictionary to store the datasets
    datasets = dict()

    # ----------- 1. Load the datasets -----------
    with catchtime() as loading_time:
        # Load train dataset (mandatory)
        datasets["train_dataset"] = load_datasets(
            dataset_locations=dataset_locations,
            datasets_to_load=config_to_execute.train_dataset,
            label_columns=config_to_execute.extra.label_columns,
            features=config_to_execute.extra.in_use_features,
            dataset_structure=config_to_execute.dataset_structure,
        )
        # Load validation dataset (optional)
        if config_to_execute.validation_dataset:
            datasets["validation_dataset"] = load_datasets(
                dataset_locations=dataset_locations,
                datasets_to_load=config_to_execute.validation_dataset,
                label_columns=config_to_execute.extra.label_columns,
                features=config_to_execute.extra.in_use_features,
                dataset_structure=config_to_execute.dataset_structure,
            )

        # Load test dataset (mandatory)
        datasets["test_dataset"] = load_datasets(
            dataset_locations=dataset_locations,
            datasets_to_load=config_to_execute.test_dataset,
            label_columns=config_to_execute.extra.label_columns,
            features=config_to_execute.extra.in_use_features,
            dataset_structure=config_to_execute.dataset_structure,
        )

        # Load reducer dataset (optional)
        if config_to_execute.reducer_dataset:
            datasets["reducer_dataset"] = load_datasets(
                dataset_locations=dataset_locations,
                datasets_to_load=config_to_execute.reducer_dataset,
                label_columns=config_to_execute.extra.label_columns_reducer,
                features=config_to_execute.extra.in_use_features,
                dataset_structure=None,
            )

        if config_to_execute.reducer_validation_dataset:
            datasets["reducer_validation_dataset"] = load_datasets(
                dataset_locations=dataset_locations,
                datasets_to_load=config_to_execute.reducer_validation_dataset,
                label_columns=config_to_execute.extra.label_columns_reducer,
                features=config_to_execute.extra.in_use_features,
                dataset_structure=None,
            )

    # Add some meta information
    additional_info["load_time"] = float(loading_time)
    additional_info["train_size"] = len(datasets["train_dataset"])
    additional_info["validation_size"] = (
        len(datasets["validation_dataset"]) if "validation_dataset" in datasets else 0
    )
    additional_info["test_size"] = len(datasets["test_dataset"])
    additional_info["reduce_size"] = (
        len(datasets["reducer_dataset"]) if "reducer_dataset" in datasets else 0
    )

    # ----------- 2. Do the non-parametric transform on train, test and reducer datasets ------------

    with catchtime() as transform_time:
        # Is there any transform to do to the datasets?
        if config_to_execute.transforms is not None:
            # Apply the transform
            datasets = do_transform(
                datasets=datasets,
                transform_configs=config_to_execute.transforms,
                keep_suffixes=True,
            )
    additional_info["transform_time"] = float(transform_time)

    # ----------- 3. Do the parametric transform on train and test, using the reducer dataset to fit the transform ------------

    with catchtime() as reduce_time:
        # Is there any reducer object and the reducer dataset is specified?
        if config_to_execute.reducer is not None:
            datasets = do_reduce(
                datasets=datasets,
                reducer_config=config_to_execute.reducer,
                reduce_on=config_to_execute.extra.reduce_on,
                use_y=config_to_execute.reducer.use_y,
            )
    additional_info["reduce_time"] = float(reduce_time)

    # ----------- 4. Do the scaling on train and test, using the train dataset to fit the scaler ------------

    with catchtime() as scaling_time:
        # Is there any scaler to do?
        if config_to_execute.scaler is not None:
            datasets = do_scaling(
                datasets=datasets,
                scaler_config=config_to_execute.scaler,
                scale_on=config_to_execute.extra.scale_on,
            )

    additional_info["scaling_time"] = float(scaling_time)

    # ----------- 5. Do the training, testing and evaluate ------------

    # Create reporter
    reporter = ClassificationReport(
        use_accuracy=True,
        use_balanced_accuracy=True,
        use_geometric_mean=True,
        use_f1_score=True,
        use_classification_report=True,
        use_confusion_matrix=True,
        plot_confusion_matrix=False,
    )

    # Run all estimators
    all_results = [
        do_classification(
            datasets=datasets,
            estimator_config=estimator_cfg,
            reporter=reporter,
        )
        for estimator_cfg in config_to_execute.estimators
    ]

    # Add some meta information
    end_time = time.time()
    additional_info["total_time"] = end_time - start_time
    additional_info["start_time"] = start_time
    additional_info["end_time"] = end_time
    additional_info["system"] = get_sys_info()

    values = {
        "experiment": asdict(config_to_execute),
        "report": all_results,
        "additional": additional_info,
    }

    # ----------- 6. Save results ------------
    with experiment_output_file.open("w") as f:
        yaml.dump(values, f, indent=4, sort_keys=True)

    # Save data if requested
    if config_to_execute.extra.save_data:
        output_dir = Path(f"{output_path}/transformed_data/{experiment_output_file.stem}")
        # Check if the folder exists
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        for dset_name, dset in datasets.items():
            file_name = (
                "train"
                if dset_name == "train_dataset"
                else "validation"
                if dset_name == "validation_dataset"
                else "test"
                if dset_name == "test_dataset"
                else "reducer_dataset"
                if dset_name == "train_reducer"
                else "validation_reducer"
            )
            # Save the data as pkl
            with open(f"{output_dir}/{file_name}.pkl", "wb") as f:
                pickle.dump(dset, f)
    return values


def run_wrapper(args) -> dict:
    """Run a single experiment. This is the function that is parallelized, if needed.
    It is a wrapper around run_experiment, that takes the arguments as a list.

    Parameters
    ----------
    args : _type_
        A list of arguments, in the following order:
        - dataset_locations: Dict[str, PathLike] (locations of the datasets)
        - output_dir: Path (the directory where the results will be stored)
        - yaml_config_file: Path (the path to the yaml file containing the experiment configuration)

    Returns
    -------
    dict
        A dict with the results of the experiment and additional information.
    """
    # Unpack arguments
    dataset_locations: Dict[str, PathLike] = args[0]
    output_dir: Path = Path(args[1])
    yaml_config_file: Path = Path(args[2])
    experiment_id = yaml_config_file.stem
    result = None
    try:
        # Load config
        config = from_dict(data_class=ExecutionConfig, data=load_yaml(yaml_config_file))
        # Create output file
        experiment_output_file = output_dir / f"{experiment_id}.yaml"
        logging.info(
            f"Starting execution {experiment_id}. Output at {experiment_output_file}"
        )

        # Run experiment
        result = run_experiment(dataset_locations, experiment_output_file, config)
    except Exception as e:
        logging.exception(f"Error while running experiment: {yaml_config_file}")
    finally:
        return result


def run_single_thread(
    args: Any,
    dataset_locations: Dict[str, PathLike],
    execution_config_files: List[PathLike],
    output_path: PathLike,
):
    """Runs the experiments sequentially, without parallelization.

    Parameters
    ----------
    args : Any
        The arguments passed to the script
    dataset_locations: Dict[str, PathLike]
        A dictionary with the dataset names and their locations.
    execution_config_files : List[PathLike]
        List of configuration files to execute.
    output_path : PathLike
        Output path where the results will be stored.
    """
    results = []
    for e in tqdm.tqdm(execution_config_files, desc="Executing experiments"):
        r = run_wrapper((dataset_locations, output_path, e))
        results.append(r)
    return results


def run_ray(
    args: Any,
    dataset_locations: Dict[str, PathLike],
    execution_config_files: List[PathLike],
    output_path: PathLike,
):
    """Runs the experiments in parallel, using Ray.

    Parameters
    ----------
    args : Any
        The arguments passed to the script
    dataset_locations: Dict[str, PathLike]
        A dictionary with the dataset names and their locations.
    execution_config_files : List[PathLike]
        List of configuration files to execute.
    output_path : PathLike
        Output path where the results will be stored.
    """
    ray.init(args.address)
    remote_func = ray.remote(run_wrapper)
    futures = [
        remote_func.remote((dataset_locations, output_path, e))
        for e in execution_config_files
    ]
    ready, not_ready = ray.wait(futures, num_returns=len(futures))

    # pool = Pool()
    # iterator = pool.imap(
    #     run_wrapper,
    #     [(dataset_locations, output_path, e) for e in execution_config_files],
    # )
    # results = list(
    #     tqdm.tqdm(
    #         iterator, total=len(execution_config_files), desc="Executing experiments"
    #     )
    # )
    return ready


if __name__ == "__main__":
    # ray.init(address="192.168.15.97:6379")
    parser = argparse.ArgumentParser(
        # prog="Execute experiments in datasets",
        description="Runs experiments in a dataset with a set of configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "execution_configs_dir",
        action="store",
        help="Directory with execution configuration files in YAML format",
        type=str,
    )

    parser.add_argument(
        "--run-name",
        action="store",
        default="execution",
        help="Name of the execution run. It will create a folder inside output dir "
        + "with this name and results will be placed inside. "
        + "Useful to run multiple executions with different configurations",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--data-path",
        action="store",
        help="Root data dir where the datasets are stored",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-l",
        "--dataset-locations",
        action="store",
        help="Dataset locations YAML file",
        type=str,
        required=False,
        default="./dataset_locations.yaml",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        default="./results",
        action="store",
        help="Output path to store results",
        type=str,
    )

    parser.add_argument(
        "--ray",
        action="store_true",
        help="Run using ray (parallel/distributed execution)",
    )

    parser.add_argument(
        "--address",
        action="store",
        default=None,
        help="Ray head node address (cluster). A local cluster will be started if nothing is informed",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip executions that were already run, that is, have something in the output path",
    )

    parser.add_argument(
        "--start",
        default=None,
        help="Number of execution config to start",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--end",
        default=None,
        help="Number of execution config to end",
        type=int,
        required=False,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Enable verbosity. Multiples -v increase level: 1=INFO, 2=Debug",
        default=0,
    )

    args = parser.parse_args()
    print(args)

    # ------ Enable logging ------
    log_level = logging.WARNING
    log_format = "[%(asctime)s] [%(hostname)s] [%(name)s] [%(levelname)s]: %(message)s"
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    coloredlogs.install(
        level=log_level, fmt=log_format, encoding="utf-8", milliseconds=True
    )

    # ------ Create output path ------
    output_path = Path(args.output_path) / args.run_name
    output_path.mkdir(parents=True, exist_ok=True)

    # ------ Load dataset locations ------
    data_path = Path(args.data_path)
    dataset_locations = load_yaml(args.dataset_locations)
    for k, v in dataset_locations.items():
        dataset_locations[k] = data_path / v

    # ------ Read and filter execution configs ------
    # Load configs from directory (sorted)
    execution_config_files = sorted(
        list(Path(args.execution_configs_dir).glob("*.yaml"))
    )
    logging.info(f"There are {len(execution_config_files)} configs (total)!")

    # Filter configs
    exp_from = args.start or 0
    exp_to = args.end or len(execution_config_files)
    execution_config_files = execution_config_files[exp_from:exp_to]

    # Skip existing?
    if args.skip_existing:
        # Calculate the difference between the execution configs and the output files (configs already executed)
        # Note, here we assume that the execution id is the same as the output file name
        to_keep_execution_ids = set(
            [e.stem for e in execution_config_files]
        ).difference(set([o.stem for o in output_path.glob("*.yaml")]))
        # Filter execution configs
        execution_config_files = [
            e for e in execution_config_files if e.stem in to_keep_execution_ids
        ]
    logging.info(f"There are {len(execution_config_files)} to execute!")

    # ------ Run experiments ------
    with catchtime() as total_time:
        # Run single
        if not args.ray:
            logging.warning("Running in sequential mode! (may be slow)")
            results = run_single_thread(
                args, dataset_locations, execution_config_files, output_path
            )
        else:
            logging.warning("Running using ray")
            results = run_ray(
                args, dataset_locations, execution_config_files, output_path
            )
            # ray.shutdown()

    if None in results:
        logging.error("Finished with errors!")
        print(f"\tFinished with errors! It took {float(total_time):.4f} seconds!")
        sys.exit(1)
    else:
        print(f"\tFinished without errors! It took {float(total_time):.4f} seconds!")
        sys.exit(0)
