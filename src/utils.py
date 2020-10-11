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


def clip_target(target):
    return np.clip(target, 0, 20)


def save_submission(y_pred, filename=None):
    assert np.max(y_pred) <= 20, "Some predicted values are greater than 20. Clip predictions to [0, 20] range first."
    assert np.min(y_pred) >= 0, "Some predicted values are lower than 20. Clip predictions to [0, 20] range first."

    try:
        df = pd.concat([test["ID"], pd.Series(y_pred, name="item_cnt_month")], axis=1)
    except NameError:
        test = pd.read_csv("../input/test.csv")
        df = pd.concat([test["ID"], pd.Series(y_pred, name="item_cnt_month")], axis=1)

    if filename is None:
        filename = "submission_" + dt.datetime.now().strftime("%Y%m%d_%H%M")
    elif ".csv" not in filename:
        filename = filename + ".csv"

    df.to_csv("../submissions/" + filename, index=False)
