import argparse
import json
import datetime as dt
from functools import partial
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder
from skopt import gp_minimize, space
from skopt.plots import plot_convergence
from utils import GroupTimeSeriesSplit, clip_target, save_submission


def optimize(params, param_names, X, y, folds):
    
    params = dict(zip(param_names, params))
    
    rmse_list = []
                
    for train_id, val_id in folds:

        X_train = X.iloc[train_id, :]
        y_train = y.iloc[train_id]
        X_val = X.iloc[val_id, :] 
        y_val = y.iloc[val_id]
        
        model, encoder = fit_model(X_train, y_train, X_val, y_val, **params)
        X_val = encoder.transform(X_val)
        y_pred = clip_target(model.predict(X_val))
        rmse_list.append(mean_squared_error(y_val, y_pred, squared=False))
        
    return np.mean(rmse_list)


def fit_model(X_train, y_train, X_val, y_val, **params):

    if args.model == "catboost":

        model = CatBoostRegressor(**params, loss_function="RMSE", random_state=42, use_best_model=True, task_type="GPU")
        model.fit(X_train, y_train,
                  cat_features=cat_cols,
                  early_stopping_rounds=20,
                  eval_set=(X_val, y_val),
                  plot=False)
        return model, None

    elif args.model == "xgboost":

        te = TargetEncoder(cols=cat_cols)
        te.fit(X_train, y_train)
        X_train = te.transform(X_train)
        X_val = te.transform(X_val)
        model = XGBRegressor(**params, random_state=42, verbosity=1, tree_method='gpu_hist', gpu_id=0)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train),
                            (X_val, y_val)],
                  eval_metric="rmse",
                  early_stopping_rounds=20,
                  verbose=True)
        return model, te

    else:

        raise ValueError("Invalid value passed to model. Has to be either CatBoost or XGBoost.")


def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-m", "--model",
                        dest="model", action="store", default="CatBoost", type=str.lower,
                        help="Either XGBoost or CatBoost.")
    parser.add_argument("-id", "--test_month_id",
                        dest="test_month_id", action="store", default=34, type=int,
                        help="ID of the month to be used as test set.")
    parser.add_argument("-cv", "--cv_type",
                        dest="cv_type", action="store", default=0, type=int, choices=range(1),
                        help="Cross-validation type to be used." + 
                             "0 - (default) the last month before test_month_id to be used as validation set." +
                             "1 - full cross-validation")
    parser.add_argument("-t", "--tune",
                        dest="tune", action="store_true",
                        help='Whether to tune hyperparameters.')
    parser.add_argument("-sm", "--save_model",
                        dest="save_model", action="store_true",
                        help="Whether to save the model.")
    parser.add_argument("-ss", "--save_submission",
                        dest="save_submission", action="store_true",
                        help="Whether to save the submission.")
    args = parser.parse_args()

    print(args)

    all_features = pd.read_parquet(config.DATA_FILE)
    # FIXME:
    all_features = all_features.loc[all_features["date_block_num"] > 11]

    X = all_features.loc[all_features["date_block_num"] < args.test_month_id].drop("item_cnt_month", axis=1)
    X_test = all_features.loc[all_features["date_block_num"] == args.test_month_id].drop("item_cnt_month", axis=1)
    y = all_features.loc[all_features["date_block_num"] < args.test_month_id, "item_cnt_month"]
    y = clip_target(y)
    id_features = all_features[["date_block_num", "item_id", "shop_id"]]
    del all_features

    if args.cv_type == 0:

        gtscv = GroupTimeSeriesSplit(n_splits=2)
        groups = np.select([
            X["date_block_num"] < args.test_month_id - 2,
            X["date_block_num"] < args.test_month_id - 1,
            X["date_block_num"] == args.test_month_id - 1],
            [0, 1, 2])
        folds = list(gtscv.split(X, y, groups=groups))[1:]

    elif args.cv_type == 1:

        gtscv = GroupTimeSeriesSplit(n_splits=33)
        folds = gtscv.split(X, y, groups=X["date_block_num"])

    if args.tune:

        if args.model == "catboost":

            param_space = [
                space.Integer(100, 5000, name="iterations"),
                space.Real(0.01, 0.1, prior="uniform", name="learning_rate"),
                space.Integer(4, 12, name="depth"),
                space.Integer(2, 30, name="l2_leaf_reg"),
                space.Integer(1, 255, name="border_count"),
                space.Real(1e-2, 10, prior="log-uniform", name="random_strength"),
                space.Real(0, 1, prior="uniform", name="bagging_temperature"),
            ]

            param_names = ["iterations", "learning_rate", "depth", "l2_leaf_reg", "border_count", "random_strength", 
                           "bagging_temperature"]

            optimization_function = partial(optimize, param_names=param_names, X=X, y=y, folds=folds)
            result = gp_minimize(optimization_function, dimensions=param_space, n_calls=20, n_random_starts=10, 
                                 verbose=10)
            best_params = dict(zip(param_names, result.x))
            print(best_params)
            plot_convergence(result)
            with open(f"../models/best_params_{args.model}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w") as file:
                file.write(json.dumps(best_params, default=convert))

        elif args.model == "xgboost":

            param_space = [
                space.Integer(100, 5000, name="n_estimators"),
                space.Real(0.01, 0.1, prior="uniform", name="learning_rate"),
                space.Integer(3, 15, name="max_depth"),
                space.Integer(1, 7, name="min_child_weight"),
                space.Real(0, 1, prior="uniform", name="gamma"),
                space.Real(0.4, 1, prior="uniform", name="colsample_bytree"),
                space.Real(0.4, 1, prior="uniform", name="subsample"),
                space.Real(0.01, 10, prior="log-uniform", name="lambda"),
                space.Real(0.01, 10, prior="log-uniform", name="alpha"),
            ]

            param_names = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", 
                           "colsample_bytree", "subsample", "lambda", "alpha"]
            
            optimization_function = partial(optimize, param_names=param_names, X=X, y=y, folds=folds)
            result = gp_minimize(optimization_function, dimensions=param_space, n_calls=20, n_random_starts=10, 
                                 verbose=10)
            best_params = dict(zip(param_names, result.x))
            print(best_params)
            plot_convergence(result)
            with open(f"../models/best_params_{args.model}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w") as file:
                file.write(json.dumps(best_params, default=convert))

        else: 

            raise ValueError("Invalid value passed to model. Has to be either CatBoost or XGBoost.")
        
    else:
        best_params = {}

    cv_scores = {}
    for train_id, val_id in folds:

        X_train = X.iloc[train_id, :]
        y_train = y.iloc[train_id]
        X_val = X.iloc[val_id, :] 
        y_val = y.iloc[val_id]

        print("Current validation set: ", np.unique(groups[val_id]))

        model, encoder = fit_model(X_train, y_train, X_val, y_val, **best_params)
        if args.model == "xgboost":
            X_val = encoder.transform(X_val)
        y_pred = clip_target(model.predict(X_val))
        cv_scores[np.unique(groups[val_id])[0]] = mean_squared_error(y_val, y_pred, squared=False)
        print(f"Average CV error: {np.array([cv_scores[i] for i in cv_scores]).mean()}")

    if args.save_model:
        model.save_model(f"../models/model_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.{args.model}")

    if args.save_submission:

        if args.model == "xgboost":
            X_test = encoder.transform(X_test)
        
        id_features = id_features.loc[id_features["date_block_num"] == args.test_month_id]
        save_submission(model, X_test, id_features, adjust_with_probing=False)
