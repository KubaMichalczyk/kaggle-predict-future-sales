import numpy as np
import pandas as pd
import datetime as dt
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold


class GroupTimeSeriesSplit(_BaseKFold):
    """
    Time Series cross-validator for a variable number of observations within the time unit.
    In the kth split, it returns first k folds as train set and the (k+1)th fold as test set.
    Indices can be grouped so that they enter the CV fold together.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    """
    def __init__(self, n_splits=5, *, max_train_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set.
            Most often just a time feature.

        Yields
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_splits = self.n_splits
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        group_counts = np.unique(groups, return_counts=True)[1]
        groups = np.split(indices, np.cumsum(group_counts)[:-1])
        n_groups = _num_samples(groups)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of groups: {1}.").format(n_folds, n_groups))
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        for test_start in test_starts:
            if self.max_train_size:
                train_start = np.searchsorted(
                    np.cumsum(group_counts[:test_start][::-1])[::-1] < self.max_train_size + 1, True)
                yield (np.concatenate(groups[train_start:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))
            else:
                yield (np.concatenate(groups[:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))


def custom_skipna(arr, func):
    a, mask = np.lib.nanfunctions._replace_nan(arr, 0)
    if np.all(mask):
        res = np.nan
    else:
        res = func(a)
    return res


def clip_target(target):
    return np.clip(target, 0, 20)


def save_submission(model, X_test, id_features=None, filename=None, adjust_with_probing=False):
    
    if id_features is None:
        id_features = X_test

    test = pd.read_csv("../input/test_preprocessed.csv",
                       dtype={"ID": np.int32,
                              "shop_id": np.int8,
                              "item_id": np.int16})
    
    if filename is None:
        filename = "submission_" + dt.datetime.now().strftime("%Y%m%d_%H%M") + ".csv"
    else:
        filename.split(".")[0] + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M") + ".csv"
    
    if adjust_with_probing:
        y_pred = model.predict(X_test) + 0.28393650
    else:
        y_pred = model.predict(X_test)
    y_pred = clip_target(y_pred)
    y_pred = pd.concat([id_features["shop_id"], id_features["item_id"],
                        pd.Series(y_pred, name="item_cnt_month", index=id_features.index)], axis=1)
    
    test \
        .merge(y_pred, how="left", on=["shop_id", "item_id"]) \
        .drop(["shop_id", "item_id"], axis=1) \
        .to_csv("../submissions/" + filename, index=False)    
