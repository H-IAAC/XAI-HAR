# Python imports
from dataclasses import dataclass
from typing import List, Optional, Union

# Librep imports
from librep.base.transform import Transform
from librep.base.estimator import Estimator
from librep.config.type_definitions import ArrayLike
from librep.estimators import SVC, KNeighborsClassifier, RandomForestClassifier, OneClassSVM, IsolationForest
from librep.transforms.fft import FFT
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Let's import a linear model classifier from sklearn to use as a wrapper
from sklearn.linear_model import LogisticRegression


# Third-party imports
from umap import UMAP
# Let's import PCA, KPCA, ICA, ISOMAP, LLE and LE for the reducers
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import numpy as np

################################################################################
# Configuration classes
################################################################################

# YAML valid confuguration keys.
# The main experiment configuration class is `ExecutionConfig`

config_version: str = "1.0"


@dataclass
class WindowedConfig:
    fit_on: Optional[str]
    transform_on: Optional[str]


@dataclass
class ReducerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict]
    use_y: Optional[bool] = False


@dataclass
class TransformConfig:
    name: str
    transform: str
    kwargs: Optional[dict] = None
    windowed: Optional[WindowedConfig] = None


@dataclass
class EstimatorConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict] = None
    num_runs: Optional[int] = 1  # Number of runs (fit/predict) for the estimator


@dataclass
class ScalerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict] = None


@dataclass
class ExtraConfig:
    in_use_features: list
    reduce_on: str  # valid values: all, sensor, axis
    scale_on: str  # valid values: self, train
    label_columns_reducer: Optional[str] = "standard activity code"
    label_columns: Optional[str] = "standard activity code"
    save_data: bool = False

@dataclass
class DatasetStructure:
    type: Optional[str] = "traditional" # valid values: traditional, k-fold, user
    user_id: Optional[str] = None
    fold: Optional[str] = None
    load_train: Optional[bool] = True
    load_validation: Optional[bool] = True
    load_test: Optional[bool] = True

@dataclass
class ExecutionConfig:
    # control variables
    version: str
    # Datasets to use to reducer
    reducer_dataset: Optional[List[str]]
    # Datasets to use to train the estimators
    train_dataset: List[str]
    test_dataset: List[str]
    # List of transforms to apply
    transforms: Optional[List[TransformConfig]]
    # The reducer to use
    reducer: Optional[ReducerConfig]
    # The Scaler
    scaler: Optional[ScalerConfig]
    # Estimator
    estimators: List[EstimatorConfig]
    # Extra
    extra: ExtraConfig
    # Dataset structure
    dataset_structure: Optional[DatasetStructure]

    # Optional ones
    reducer_validation_dataset: Optional[List[str]] = None
    validation_dataset: Optional[List[str]] = None

################################################################################
# Transforms
################################################################################
class Identity(Transform):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def transform(self, X):
        return X


class WrapperEstimatorTransform(Estimator, Transform):
    def __init__(self, obj: Union[Estimator, Transform], *args, **kwargs) -> None:
        self._obj = obj

    def predict(self, X):
        return self._obj.predict(X)

    def transform(self, X):
        return self._obj.transform(X)


class DebugTransformEstimator(Transform, Estimator):
    def __init__(self, *args, **kwargs) -> None:
        print(f"DebugTransformEstimator: args={args}, kwargs={kwargs}")

    def fit(self, X, y=None, X_val=None, y_val=None, **fit_params):
        print(
            f"DebugTransformEstimator (fit). X: {len(X)}, y: {len(y) if y is not None else None}, X_val: {len(X_val) if X_val is not None else None}, y_val: {len(y_val) if y_val is not None else None}, fit_params: {fit_params}"
        )
        return self

    def transform(self, X):
        print(f"DebugTransformEstimator (transform): {X.shape}")
        return X
    
    def predict(self, X):
        print(f"DebugTransformEstimator (predict): {X.shape}")
        return np.zeros(len(X))


################################################################################
# Constants (Valid keys)
################################################################################

# Dictionary with the valid estimators keys to use in experiment configuration
# (under estimator.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Estimators must be a subclass of `librep.estimators.base.BaseEstimator` or implement
# the same interface (scikit-learn compatible, fit/predict methods)
estimator_cls = {
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "DebugEstimator": DebugTransformEstimator,
    "WrapperEstimator": WrapperEstimatorTransform,
    "OC-SVM": OneClassSVM,
    "IsolationForest": IsolationForest,
}

# Dictionary with the valid reducer keys to use in experiment configuration
# (under reducer.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Reducers must be a subclass of `librep.reducers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
reducers_cls = {
    "identity": Identity,
    "umap": UMAP,
    "pca": PCA,
    "kernel_pca": KernelPCA,
    "isomap": Isomap,
    "lle": LocallyLinearEmbedding,
    "ica": FastICA,
    "WrapperTransform": WrapperEstimatorTransform,
    "DebugReducer": DebugTransformEstimator,
}

# Dictionary with the valid transforms keys to use in experiment configuration
# (under transform.transform key).
# The key is the algorithm name and the value is the class to use.
# Transforms must be a subclass of `librep.transforms.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
transforms_cls = {
    "identity": Identity,
    "fft": FFT,
    "WrapperTransform": WrapperEstimatorTransform,
    "DebugTransform": DebugTransformEstimator
}

# Dictionary with the valid scalers keys to use in experiment configuration
# (under scaler.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Scalers must be a subclass of `librep.scalers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
scaler_cls = {
    "identity": Identity,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}

# Dictionary with standard labels for each activity code
standard_labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}
