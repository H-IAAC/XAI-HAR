from typing import Callable, List, Union, Any
import numpy as np

import pandas as pd
import scipy
from scipy import signal, interpolate
from scipy import constants
import tqdm
import plotly.express as px

from typing import Tuple
import random

"""This module contains the classes that will be used to transform the data. The classes are callable objects, that is, they implement the __call__ method.
The __call__ method receives a dataframe as a parameter and returns a dataframe.
The classes are used to create a pipeline of transformations, which will be applied in the order they were added.
The pipeline is created using the Pipeline class, which receives a list of transformations as a parameter.
The transformations must be callable objects, that is, that implement the __call__ method.
The __call__ method must receive a dataframe as a parameter and return a dataframe.
"""                                                      

class FilterByCommonRows:
    """Filter the dataframe to only have rows that are present in both dataframes."""

    def __init__(self, match_columns: Union[str, List[str]]):
        self.match_columns = (
            match_columns if isinstance(match_columns, list) else [match_columns]
        )
        """Filter the dataframe to only have rows that are present in both dataframes.

        Parameters
        ----------
        match_columns : Union[str, List[str]]
            Name of the column(s) to be used to filter the dataframe.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The filtered dataframes.
        """

    def __call__(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter the dataframe to only have rows that are present in both dataframes.

        Parameters
        ----------
        df1 : pd.DataFrame
            First dataframe to be filtered.
        df2 : pd.DataFrame
            Second dataframe to be filtered.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The filtered dataframes.
        """
        common_rows = set(
            df1[self.match_columns].itertuples(index=False, name=None)
        ) & set(df2[self.match_columns].itertuples(index=False, name=None))
        df1_filtered = df1[
            df1[self.match_columns].apply(tuple, axis=1).isin(common_rows)
        ]
        df2_filtered = df2[
            df2[self.match_columns].apply(tuple, axis=1).isin(common_rows)
        ]
        return df1_filtered, df2_filtered
    
class SplitGuaranteeingAllClassesPerSplit:
    """Split the dataframe in a way that all classes are present in both splits."""

    def __init__(
        self,
        column_to_split: str = "user",
        class_column: str = "standard activity code",
        train_size: float = 0.8,
        random_state: int = 42,
        retries: int = 10,
    ):
        """ "
        Parameters
        ----------
        column_to_split : str, optional
            Name of the column to be used to split the dataframe, by default "user"
        class_column : str, optional
            Name of the column that contains the class, by default "standard activity code"
        train_size : float, optional
            Percentage of the dataframe that will be used for training, by default 0.8
        random_state : int, optional
            Random state to be used, by default None
        retries : int, optional
            Number of retries to be used, by default 10

        Raises
        ------
        ValueError
            If it is not possible to split the dataframe in a way that all classes are present in both splits.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The filtered dataframes.
        """
        self.column_to_split = column_to_split
        self.class_column = class_column
        self.train_size = train_size
        self.random_state = random_state
        self.retries = retries

    def __call__(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe in a way that all classes are present in both splits.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to be splitted.

        Raises
        ------
        ValueError
            If it is not possible to split the dataframe in a way that all classes are present in both splits.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The filtered dataframes.
        """

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        split_values = dataframe[self.column_to_split].unique()  # user ids
        class_values = dataframe[self.class_column].unique()  # activity codes
        for retrie in range(self.retries):
            # Let's shuffle the values and split them with a fixed random state to guarantee that the split is always the same
            random.seed(self.random_state)
            random.shuffle(split_values)

            train_values = split_values[: int(len(split_values) * self.train_size)]
            test_values = split_values[int(len(split_values) * self.train_size) :]

            train_df = dataframe.loc[dataframe[self.column_to_split].isin(train_values)]
            test_df = dataframe.loc[dataframe[self.column_to_split].isin(test_values)]

            if len(train_df[self.class_column].unique()) != len(class_values):
                random.seed(self.random_state + retrie)
                continue
            if len(test_df[self.class_column].unique()) != len(class_values):
                random.seed(self.random_state + retrie)
                continue
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

        raise ValueError(
            "Could not split dataframe in a way that all classes are present in both splits"
        )

class SplitGuaranteeingAllUsersPerSplit:

    def __init__(
        self,
        column_to_split: str = "user",
        train_size: float = 0.8,
        random_state: int = 42,
        retries: int = 10,
    ):     

        self.column_to_split = column_to_split
        self.train_size = train_size
        self.random_state = random_state
        self.retries = retries

    def __call__(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe in a way that all users are present in both splits.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to be splitted.

        Raises
        ------
        ValueError
            If it is not possible to split the dataframe in a way that all users are present in both splits.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The filtered dataframes.
        """

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        users = list(dataframe[self.column_to_split].unique())  # user ids

        n_samples_per_user = {
            user: len(dataframe.loc[dataframe[self.column_to_split] == user])
            for user in users
        }

        dfs_train = []
        dfs_test = []
        for user, n_samples in n_samples_per_user.items():
            df = dataframe.loc[dataframe[self.column_to_split] == user]
            random.seed(self.random_state)

            n_samples_to_train = int(n_samples * self.train_size)
            n_samples_to_test = n_samples - n_samples_to_train

            if n_samples_to_train == 0 or n_samples_to_test == 0:
                random.seed(self.random_state)
                continue
            else:
                train_df = df.sample(n_samples_to_train, random_state=self.random_state)
                test_df = df.drop(train_df.index)
                dfs_train.append(train_df)
                dfs_test.append(test_df)
            
        train_df = pd.concat(dfs_train)
        test_df = pd.concat(dfs_test)
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

class BalanceToMinimumClassAndUser:
    """Balance the dataframe to the minimum class size per user. User without a minimum class size will be discarded."""

    def __init__(
        self,
        class_column: str = "standard activity code",
        filter_column: str = "user",
        random_state: int = 42,
        min_value: int = None,
    ):
        """
        Parameters
        ----------
        class_column : str, optional
            Name of the column that contains the class, by default "standard activity code"
        filter_column : str, optional
            Name of the column that contains the user, by default "user"
        random_state : int, optional
            Random state to be used, by default 42
        min_value : int, optional
            Minimum size of the class, by default None
        """
        self.class_column = class_column
        self.random_state = random_state
        self.min_value = min_value
        self.filter_column = filter_column

    def __call__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataframe to the minimum class size per user. User without a minimum class size will be discarded.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to be balanced.

        Returns
        -------
        pd.DataFrame
            Balanced dataframe.
        """

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        class_values = dataframe[self.class_column].unique()
        min_value_size = self.min_value

        # First we need to filter the dataframe to only have filter column values that are present in all classes
        filter_values = dataframe[self.filter_column].unique()
        filter_values_to_use = []

        for filter_value in filter_values:
            if sorted(
                dataframe[dataframe[self.filter_column] == filter_value][
                    self.class_column
                ].unique()
            ) == sorted(class_values):
                filter_values_to_use.append(filter_value)
        df = dataframe[dataframe[self.filter_column].isin(filter_values_to_use)].copy()

        # Now we can balance the dataframe
        if self.min_value is None:
            min_value_size = min(
                [
                    len(
                        df.loc[
                            (df[self.class_column] == class_value)
                            & (df[self.filter_column] == filter_value)
                        ]
                    )
                    for class_value in class_values
                    for filter_value in filter_values_to_use
                ]
            )

        balanced_df = pd.concat(
            [
                df.loc[
                    (df[self.class_column] == class_value)
                    & (df[self.filter_column] == filter_value)
                ].sample(min_value_size, random_state=self.random_state)
                for class_value in class_values
                for filter_value in filter_values_to_use
            ]
        )
        return balanced_df
    
class LeaveOneSubjectOut:
    """Leave one subject out cross-validation split.
    The dataframe is split in a way that all samples from a single subject are present in the test set.
    
    Parameters
    ----------
    column_to_split : str, optional
        Name of the column to be used to split the dataframe, by default "user"
    
    Returns
    -------
    Dict[Union[str, int], Tuple[pd.DataFrame, pd.DataFrame]]
        A dictionary with the splits.
    """

    def __init__(
        self,
        column_to_split: str = "user",
    ):
        """
        Parameters
        ----------
        column_to_split : str, optional
            Name of the column to be used to split the dataframe, by default "user"
        """
        self.column_to_split = column_to_split

    def __call__(self, dataframe: pd.DataFrame) -> dict:
        
        splits = dataframe[self.column_to_split].unique()

        dfs = {key: (None, None) for key in splits}

        # for split in splits:
        for i, split in enumerate(splits):
            train_df = dataframe.loc[dataframe[self.column_to_split] != split]
            if i == len(splits):
                val_df = dataframe.loc[dataframe[self.column_to_split] == splits[0]]
            else:
                val_df = dataframe.loc[dataframe[self.column_to_split] == splits[i]]
            test_df = dataframe.loc[dataframe[self.column_to_split] == split]
            # Remove users in val_df from train_df
            train_df = train_df.loc[~train_df[self.column_to_split].isin(val_df[self.column_to_split].unique())]
            dfs[split] = (train_df, val_df, test_df)

        return dfs
class GenerateFold:
    """Generate a fold for a given dataframe.
    The fold is generated in a way that all users are present in both splits.

    Parameters
    ----------
    column_to_split : str, optional
        Name of the column to be used to split the dataframe, by default "user"
    n_splits : int, optional
        Number of splits to be generated, by default 5
    random_state : int, optional
        Random state to be used, by default 42

    Raises
    ------
    ValueError
        If it is not possible to split the dataframe in a way that all users are present in both splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The filtered dataframes.    
    """

    def __init__(
        self,
        column_to_split: str = "user",
        n_folds: int = 5,
        random_state: int = 42,
    ):
        
        self.column_to_split = column_to_split
        self.n_folds = n_folds
        self.random_state = random_state

    def __call__(self, dataframe: pd.DataFrame) -> dict:

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        folders = {
            i: pd.DataFrame(columns=dataframe.columns) for i in range(self.n_folds)
        }

        spliters = list(dataframe[self.column_to_split].unique())
        frac = int(len(spliters) / self.n_folds)

        random.seed(self.random_state)
        random.shuffle(spliters)

        for i in range(self.n_folds):
            split = spliters[i*frac:(i+1)*frac]
            df = dataframe[dataframe[self.column_to_split].isin(split)].copy()
            folders[i] = df
            
        split = spliters[i*frac:]
        df = dataframe[dataframe[self.column_to_split].isin(split)].copy()
        folders[i] =  df

        return folders
            
class BalanceToMinimumClass:
    """Balance the dataframe to the minimum class size."""

    def __init__(
        self,
        class_column: str = "standard activity code",
        random_state: int = 42,
        min_value: int = None,
    ):
        """
        Parameters
        ----------
        class_column : str, optional
            Name of the column that contains the class, by default "standard activity code"
        random_state : int, optional
            Random state to be used, by default 42
        min_value : int, optional
            Minimum size of the class, by default None
        """
        self.class_column = class_column
        self.random_state = random_state
        self.min_value = min_value

    def __call__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataframe to the minimum class size.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to be balanced.

        Returns
        -------
        pd.DataFrame
            Balanced dataframe.
        """

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        class_values = dataframe[self.class_column].unique()
        min_valuse_size = self.min_value

        if min_valuse_size is None:
            min_valuse_size = min(
                [
                    len(dataframe.loc[dataframe[self.class_column] == class_value])
                    for class_value in class_values
                ]
            )
        balanced_df = pd.concat(
            [
                dataframe.loc[dataframe[self.class_column] == class_value].sample(
                    min_valuse_size, random_state=self.random_state
                )
                for class_value in class_values
            ]
        )
        return balanced_df


class Interpolate:
    """Interpolate columns of the dataframe assuming that the data is at a fixed frequency.
    Uses the `scipy.interpolate` function to interpolate the data."""

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        features_to_select: Union[str, List[str]],
        original_fs: float,
        target_fs: float,
        kind: str = "cubic",
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Name of the column(s) to be grouped to resample.
            Normally grouped by user event
            (otherwise calculates the difference using the entire dataframe, with samples from different events and users).
        features_to_select : Union[str, List[str]]
            Name of the column(s) to be resampled.
        original_fs : float
            Original sampling frequency.
        target_fs : float
            Desired sampling frequency.
        kind : str, optional
            Type of interpolation to be used, by default 'cubic'.
        """
        self.groupby_column = groupby_column
        self.features_to_select = (
            [features_to_select]
            if isinstance(features_to_select, str)
            else features_to_select
        )
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.kind = kind

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the columns of the dataframe.
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be interpolated.
        Returns
        -------
        pd.DataFrame
            The dataframe with the desired columns, interpolated.
        """
        df = df.reset_index()
        for _, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column, group_keys=True), desc="Interpoling"
        ):
            for column in self.features_to_select:
                signal = grouped_df[column].values
                arr = np.array([np.nan] * len(grouped_df))
                time = np.arange(0, len(signal), 1) / self.original_fs
                interplator = interpolate.interp1d(
                    time,
                    signal,
                    kind=self.kind,
                )
                new_time = np.arange(0, time[-1], 1 / self.target_fs)
                resampled = interplator(new_time)

                arr[: len(resampled)] = resampled
                df.loc[grouped_df.index, column] = arr
        return df.dropna().reset_index(drop=True)


class AddGravityColumn:
    """Add a column with gravity in each axis."""

    def __init__(self, axis_columns: List[str], gravity_columns: List[str]):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Name of the columns that contain the acceleration data.
        gravity_columns : List[str]
            Name of the column that contains the gravity data.
        """
        self.axis_columns = axis_columns
        self.gravity_columns = gravity_columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a column with gravity in each axis.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be used.

        Returns
        -------
        pd.DataFrame
            Dataframe with the gravity data added.
        """
        for axis_col, gravity_col in zip(self.axis_columns, self.gravity_columns):
            df[axis_col] = df[axis_col] + df[gravity_col]
        return df


class Convert_G_to_Ms2:
    """Convert the acceleration from g to m/s²."""

    def __init__(self, axis_columns: List[str], g_constant: float = constants.g):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Name of the columns that contain the acceleration data.
        g_constant : float, optional
            Value of gravity to be added, by default `scipy.constants.g`
        """
        self.axis_columns = axis_columns
        self.gravity_constant = g_constant

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the conversion from g to m/s².

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be used.

        Returns
        -------
        pd.DataFrame
            Dataframe with the converted acceleration data.
        """
        for axis_col in self.axis_columns:
            df[axis_col] = df[axis_col] * self.gravity_constant
        return df


class ButterworthFilter:
    """Apply the Butterworth filter to remove gravity."""

    def __init__(self, axis_columns: List[str], fs: float, Wn: Any = 0.3, btype: str = "hp"):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Name of the columns that contain the acceleration data.
        fs : float
            Original frequency of the dataset
        Wn : Any, optional
            Cutoff frequency, by default 0.3
        btype : str, optional
            Type of filter to be used, by default "hp"
            The available options are: {lowpass, highpass, bandpass, bandstop}
        """
        self.axis_columns = axis_columns
        self.fs = fs
        self.Wn = Wn
        self.btype = btype

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the Butterworth filter to remove gravity.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be used.

        Returns
        -------
        pd.DataFrame
            Dataframe with the filtered acceleration data (passed filter).
        """
        h = signal.butter(3, self.Wn, btype=self.btype, fs=self.fs, output="sos")
        for axis_col in self.axis_columns:
            df[axis_col] = signal.sosfiltfilt(h, df[axis_col].values)
        return df
    
class MedianFilter:
    """Apply the Median filter to remove gravity."""

    def __init__(self, axis_columns: List[str], kernel_size: int = 3):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Name of the columns that contain the acceleration data.
        kernel_size : int, optional
            Size of the kernel to be used, by default 3

        Returns
        -------
        pd.DataFrame
            Dataframe with the filtered acceleration data (passed filter).

        """
        self.axis_columns = axis_columns
        self.kernel_size = kernel_size
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the Butterworth filter to remove gravity.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be used.

        Returns
        -------
        pd.DataFrame
            Dataframe with the filtered acceleration data (passed filter).
        """
        for axis_col in self.axis_columns:
            df[axis_col] = signal.medfilt(df[axis_col].values, kernel_size=self.kernel_size)
        return df


class CalcTimeDiffMean:
    """Calc the difference between the time intervals."""

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        column_to_diff: str,
        new_column_name: str = "diff",
        filter_predicate: Callable[[pd.DataFrame], pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Name of the column(s) to be grouped to calculate the difference.
            Normally grouped by user event
            (otherwise calculates the difference using the entire dataframe, with samples from different events and users).
        column_to_diff : str
            Name of the column to be used to calculate the difference.
        new_column_name : str, optional
            Name of the column where the difference will be stored, by default "diff"
        filter_predicate : Callable[[pd.DataFrame], pd.DataFrame], optional
            Function that filters the dataframe, by default None
        """
        self.groupby_column = groupby_column
        self.column_to_diff = column_to_diff
        self.new_column_name = new_column_name
        self.filter_predicate = filter_predicate

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calc the difference between the time intervals.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be used.

        Returns
        -------
        pd.DataFrame
            Dataframe with the column with the difference between the time intervals.
            If `filter_predicate` is not None, the dataframe will be filtered.
        """
        df[self.new_column_name] = df.groupby(self.groupby_column)[
            self.column_to_diff
        ].diff()
        df = df.dropna(subset=[self.new_column_name])
        if self.filter_predicate:
            df = df.groupby(self.groupby_column).filter(self.filter_predicate)
        return df.reset_index(drop=True)


class PlotDiffMean:
    """Plot the histogram of the difference between the time intervals."""

    def __init__(self, column_to_plot: str = "diff"):
        """
        Parameters
        ----------
        column_to_plot : str, optional
            Column to be used to print the histogram, by default "diff"
        """
        self.column_to_plot = column_to_plot

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Print the histogram of the difference between the time intervals.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            The dataframe itself.
        """

        fig = px.histogram(df, x=self.column_to_plot)
        fig.show("png")
        return df


class Resampler:
    """Resample columns of the dataframe assuming that the data is at a fixed frequency.
    Uses the `scipy.signal.resample` function to resample the data.
    """

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        features_to_select: Union[str, List[str]],
        original_fs: float,
        target_fs: float,
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Name of the column(s) to be grouped to resample.
            Normally grouped by user event
            (otherwise calculates the difference using the entire dataframe, with samples from different events and users).
        features_to_select : Union[str, List[str]]
            Name of the column(s) to be resampled.
        original_fs : float
            Original sampling frequency.
        target_fs : float
            Desired sampling frequency.
        """
        self.groupby_column = groupby_column
        self.features_to_select = (
            [features_to_select]
            if isinstance(features_to_select, str)
            else features_to_select
        )
        self.original_fs = original_fs
        self.target_fs = target_fs

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the columns of the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be resampled.

        Returns
        -------
        pd.DataFrame
            The dataframe with the desired columns, resampled.
        """
        df = df.reset_index()
        for key, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column, group_keys=True), desc="Resampling"
        ):
            for column in self.features_to_select:
                time = len(grouped_df) // self.original_fs
                arr = np.array([np.nan] * len(grouped_df))
                resampled = signal.resample(
                    grouped_df[column].values, int(time * self.target_fs)
                )
                arr[: len(resampled)] = resampled
                df.loc[grouped_df.index, column] = arr
        return df.dropna().reset_index(drop=True)


class ResamplerPoly:
    """Resample columns of the dataframe assuming that the data is at a fixed frequency.
    Uses the `scipy.signal.resample_poly` function to resample the data.
    """

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        features_to_select: Union[str, List[str]],
        up: float,
        down: float,
        padtype: str = "mean",
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Name of the column(s) to be grouped to resample.
            Normally grouped by user event
            (otherwise calculates the difference using the entire dataframe, with samples from different events and users).
        features_to_select : Union[str, List[str]]
            Name of the column(s) to be resampled.
        up : float
            Increase factor of the frequency.
        down : float
            Frequency reduction factor.
        padtype : str, optional
            Type of padding, by default 'mean'.
        """
        self.groupby_column = groupby_column
        self.features_to_select = (
            [features_to_select]
            if isinstance(features_to_select, str)
            else features_to_select
        )
        self.up = up
        self.down = down
        self.padtype = padtype

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample the columns of the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be resampled.

        Returns
        -------
        pd.DataFrame
            The dataframe with the desired columns, resampled.
        """
        df = df.reset_index()
        for _, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column, group_keys=True), desc="Resampling"
        ):
            for column in self.features_to_select:
                arr = np.array([np.nan] * len(grouped_df))
                resampled = signal.resample_poly(
                    grouped_df[column].values,
                    up=self.up,
                    down=self.down,
                    padtype=self.padtype,
                )
                arr[: len(resampled)] = resampled
                df.loc[grouped_df.index, column] = arr
        return df.dropna().reset_index(drop=True)


class Windowize:
    """Realize the windowing of the data in fixed size windows.
    The windowing will be done with consecutive samples of the dataframe and the last window will be discarded.
    The desired columns will be transposed (from row to column) in the desired window size.
    For the remaining columns, the first element of the window will be kept.
    Note: it is assumed here that the window has no overlap and that the sampling rate is constant.
    """

    def __init__(
        self,
        features_to_select: List[str],
        samples_per_window: int,
        samples_per_overlap: int,
        groupby_column: Union[str, List[str]],
        divisible_by: int = None,
    ):
        """
        Parameters
        ----------
        features_to_select : List[str]
            Features that will be used to perform the windowing
            (will be transposed from rows to columns and a suffix of index will be added).
        samples_per_window : int
            Number of consecutive samples that will be used to perform the windowing.
        samples_per_overlap : int
            Number of samples that will be overlapped between consecutive windows.
        groupby_column : Union[str, List[str]]
            Name of the column(s) to be grouped to perform the windowing.
            Normally grouped by user event
            (otherwise calculates the difference using the entire dataframe, with samples from different events and users).
        """
        self.features_to_select = (
            features_to_select
            if isinstance(features_to_select, list)
            else [features_to_select]
        )
        self.samples_per_window = samples_per_window
        self.samples_per_overlap = samples_per_overlap
        self.groupby_column = groupby_column
        self.divisible_by = divisible_by

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform the windowing on the columns of the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be windowed.

        Returns
        -------
        pd.DataFrame
            The dataframe with fixed size windows.
        """
        values = []
        other_columns = set(df.columns) - set(self.features_to_select)

        for key, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column), desc="Creating windows"
        ):
            index = 0
            for i, start in enumerate(
                range(
                    0,
                    len(grouped_df),
                    self.samples_per_window - self.samples_per_overlap,
                )
            ):
                window_df = grouped_df[
                    start : start + self.samples_per_window
                ].reset_index(drop=True)
                if len(window_df) != self.samples_per_window:
                    continue
                if window_df.isnull().values.any():
                    continue

                features = window_df[self.features_to_select].unstack()
                features.index = features.index.map(
                    lambda a: f"{a[0]}-{(a[1])%(self.samples_per_window)}"
                )
                for column in other_columns:
                    features[column] = window_df[column].iloc[0]
                features["window"] = index
                index += 1
                values.append(features)
        return pd.concat(values, axis=1).T.reset_index(drop=True)
    

class Peak_Windowize:
    """Realize the windowing of the data in fixed size windows.
    The windowing will be done centered in the peak of the smv column. 
    In case of a random peak, the window will be shifted randomly.
    If the beginning or end of the window is out of the dataframe, the window will be shifted to the left or right.
    The desired columns will be transposed (from row to column) in the desired window size.
    For the remaining columns, the first element of the window will be kept.
    Note: it is assumed here that the window has no overlap and that the sampling rate is constant.
    """

    def __init__(
        self,
        features_to_select: List[str],
        samples_per_window: int,
        groupby_column: Union[str, List[str]],
        peak: str = "random", # "random", "max",
        shift: int = 5,

    ):
        """
        Parameters
        ----------
        features_to_select : List[str]
            Features that will be used to perform the windowing
            (will be transposed from rows to columns and a suffix of index will be added).
        samples_per_window : int
            Number of consecutive samples that will be used to perform the windowing.
        groupby_column : Union[str, List[str]]
            Name of the column(s) to be grouped to perform the windowing.
            Normally grouped by user event
            (otherwise calculates the difference using the entire dataframe, with samples from different events and users).
        peak : str
            Type of peak to be used to center the window.
        """

        self.features_to_select = (
            features_to_select
            if isinstance(features_to_select, list)
            else [features_to_select]
        )
        self.samples_per_window = samples_per_window
        self.groupby_column = groupby_column
        self.peak = peak
        self.shift = shift

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform the windowing on the columns of the dataframe with the peak in the center of the window.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be windowed.

        Returns
        -------
        pd.DataFrame
            The dataframe with fixed size windows.
        """

        values = []
        other_columns = set(df.columns) - set(self.features_to_select)

        for key, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column), desc="Creating windows based on peak"
        ):
            
            window_df = grouped_df[self.features_to_select].reset_index(drop=True)
            smv_peak = window_df["smv"].max()
            if self.peak == "random":
                center = window_df["smv"].idxmax()
                center = random.randint(center - self.shift, center + self.shift)
            elif self.peak == "max":
                center = window_df["smv"].idxmax()
            else:
                raise ValueError("Peak must be 'random' or 'max'")
            
            start = center - self.samples_per_window // 2
            end = center + self.samples_per_window // 2

            if start < 0:
                end = end - start
                start = 0

            if end > len(window_df):
                start = start - (end - len(window_df))
                end = len(window_df)


            window_df = window_df[start:end].reset_index(drop=True)

            if len(window_df) != self.samples_per_window:
                continue
            if window_df.isnull().values.any():
                continue

            features = window_df[self.features_to_select].unstack()
            features.index = features.index.map(
                lambda a: f"{a[0]}-{(a[1])%(self.samples_per_window)}"
            )
            for column in other_columns:
                features[column] = grouped_df[column].iloc[0]
            values.append(features)
        return pd.concat(values, axis=1).T.reset_index(drop=True)

class AddStandardActivityCode:
    """Add the column "standard activity code" to the dataframe."""

    def __init__(self, codes_map: dict):
        """
        Parameters
        ----------
        codes_map : dict
            Dictionary with the activity code (from the original dataset)
            as key and the standard activity code as value
        """
        self.codes_map = codes_map

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the column "standard activity code" to the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be added to the column.

        Returns
        -------
        pd.DataFrame
            The dataframe with the column "standard activity code" added.
        """
        df["standard activity code"] = df["activity code"].map(self.codes_map)
        return df

class AddSMV:
    """Add the column "smv" to the dataframe."""

    def __init__(self, axis_columns: List[str]):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Name of the columns that contain the acceleration data.
        """
        self.axis_columns = axis_columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the column "smv" to the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be added to the column.

        Returns
        -------
        pd.DataFrame
            The dataframe with the column "smv" added.
        """
        df["smv"] = np.sqrt(
            df[self.axis_columns[0]] ** 2
            + df[self.axis_columns[1]] ** 2
            + df[self.axis_columns[2]] ** 2
        )
        return df

class RenameColumns:
    """Rename dataframe columns."""

    def __init__(self, columns_map: dict):
        """
        Parameters
        ----------
        columns_map : dict
            Dictionary with the original column names as key and the new column name as value.
        """

        self.columns_map = columns_map

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename dataframe columns.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with the columns to be renamed.

        Returns
        -------
        pd.DataFrame
            The dataframe with the renamed columns.
        """

        df.rename(columns=self.columns_map, inplace=True)
        return df


class Pipeline:
    """Data transformation pipeline."""

    def __init__(self, transforms: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        Parameters
        ----------
        transforms : Callable[[pd.DataFrame], pd.DataFrame]
            List of transformations to be executed.
            The transformations must be callable objects, that is, that implement the __call__ method.
            The __call__ method must receive a dataframe as a parameter and return a dataframe.
        """
        self.transforms = transforms

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformations in the order they were added.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        for transform in self.transforms:
            print(f"Executing {transform.__class__.__qualname__}")
            df = transform(df)
        return df
