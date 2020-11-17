import numpy as np
import pandas as pd
import itertools
from read_data import read_data


def expand_shop_item_grid(sales_train_by_month):

    group_dfs = []
    for group_id, group_df in sales_train_by_month.groupby("date"):
        group_dfs.append(
            pd.DataFrame(itertools.product(group_df["shop_id"].unique(),
                                           group_df["item_id"].unique(),
                                           group_df["date"].unique(),
                                           group_df["date_block_num"].unique()),
                         columns=["shop_id", "item_id", "date", "date_block_num"]) \
                .merge(group_df, how="left", on=["shop_id", "item_id", "date", "date_block_num"]))
    expanded_df = pd.concat(group_dfs, axis=0)

    return expanded_df


def drop_duplicated_shops(shops):
    """As shown in EDA, some shops were probably listed under two different IDs (possibly because of slightly different
    names). This function alters their ID to identify appropriate pairs under a single ID. """

    shop_id_mapping = dict(zip(shops["shop_id"].unique(), shops["shop_id"].unique()))
    shop_id_mapping[11] = 10
    shop_id_mapping[0] = 57
    shop_id_mapping[1] = 58

    shops["shop_id"] = shops["shop_id"].map(shop_id_mapping)
    shops = shops.drop_duplicates(subset="shop_id", keep="last").sort_values("shop_id").reset_index(drop=True)

    return shops, shop_id_mapping


def read_preprocessed_data():
    sales_by_month = pd.read_csv("../input/sales_by_month.csv",
                                 dtype={"date_block_num": np.int8,
                                        "shop_id": np.int8,
                                        "item_id": np.int16,
                                        "mean_item_price": np.float32,
                                        "median_item_price": np.float32,
                                        "item_cnt_month": np.float32,
                                        "ID": np.float32,
                                        "item_category_id": np.int8},
                                 parse_dates=["date"])

    sales_train = pd.read_csv("../input/sales_train_preprocessed.csv",
                              dtype={"date_block_num": np.int8,
                                      "shop_id": np.int8,
                                      "item_id": np.int16,
                                      "item_price": np.float32,
                                      "item_cnt_day": np.int32},
                              parse_dates=["date"],
                              dayfirst=True)

    test = pd.read_csv("../input/test_preprocessed.csv",
                       dtype={"ID": np.int32,
                              "shop_id": np.int8,
                              "item_id": np.int16})

    items = pd.read_csv("../input/items.csv",
                        dtype={"item_id": np.int16,
                               "item_category_id": np.int8})

    item_categories = pd.read_csv("../input/item_categories.csv",
                                  dtype={"item_category_id": np.int8})

    shops = pd.read_csv("../input/shops_preprocessed.csv",
                        dtype={"shop_id": np.int8})

    calendar = pd.read_csv("../auxiliaries/bank_holidays_calendar.csv",
                           dtype={"weekday": np.int8},
                           parse_dates=["date"])

    return sales_by_month, sales_train, test, items, item_categories, shops, calendar


if __name__ == "__main__":

    sales_train, test, items, item_categories, shops, calendar = read_data()

    # TODO: Test if it leads to better model performance or should we leave shop_id unchanged
    shops, shop_id_mapping = drop_duplicated_shops(shops)
    sales_train["shop_id"] = sales_train["shop_id"].map(shop_id_mapping)
    test["shop_id"] = test["shop_id"].map(shop_id_mapping)

    sales_train.loc[sales_train["item_price"] < 0, "item_price"] = np.nan

    sales_train_by_month = sales_train \
        .groupby(["shop_id", "item_id", "date_block_num", sales_train["date"].dt.to_period("M").dt.to_timestamp()]) \
        [["item_price", "item_cnt_day"]] \
        .agg({"item_price": [np.mean, np.median], "item_cnt_day": np.sum}) \
        .reset_index()
    sales_train_by_month.columns = ["_".join(col[::-1]).strip("_") for col in sales_train_by_month.columns]
    sales_train_by_month.rename({"sum_item_cnt_day": "item_cnt_month"}, axis=1, inplace=True)

    sales_train_by_month = expand_shop_item_grid(sales_train_by_month)
    # TODO: Test if a different imputation method (e.g. KNN) will lead to a better performance
    sales_train_by_month["item_cnt_month"] = sales_train_by_month["item_cnt_month"].fillna(0)

    test["date"] = pd.to_datetime("2015-11-01")
    test["date_block_num"] = 34
    test["date_block_num"] = test["date_block_num"].astype(np.int8)
    sales_by_month = pd.concat([sales_train_by_month, test], axis=0, ignore_index=True)
    test.drop(["date", "date_block_num"], axis=1, inplace=True)

    sales_by_month = sales_by_month \
        .merge(items[["item_id", "item_category_id"]], on="item_id", how="left")

    sales_by_month.sort_values(["date", "shop_id", "item_id"], inplace=True)

    sales_by_month.to_csv("../input/sales_by_month.csv", index=False)
    sales_train.to_csv("../input/sales_train_preprocessed.csv", index=False)
    test.to_csv("../input/test_preprocessed.csv", index=False)
    shops.to_csv("../input/shops_preprocessed.csv", index=False)