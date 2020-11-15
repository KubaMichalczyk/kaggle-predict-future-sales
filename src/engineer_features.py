import pandas as pd
import numpy as np
import translate
from read_data import read_data
from preprocess_data import read_preprocessed_data
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from utils import GroupTimeSeriesSplit
from functools import wraps


def reindex(base_extract_fn):
    """
    A decorator that takes a function creating mapping_df (a helper data frame with additional information) and
    reindex mapping_df values to create features compatible with main_df index.
    """
    @wraps(base_extract_fn)
    def wrapper(main_df, mapping_df, on, **kwargs):
        mapping_df = base_extract_fn(mapping_df, **kwargs)
        cols = mapping_df.columns.drop(on)
        return main_df.merge(mapping_df, how="left", on=on)[cols]
    return wrapper


@reindex
def extract_shop_features(shops):

    cities = shops["shop_name"].str.extract("([A-Яa-я]+\\.[A-Яa-я]+)|([A-Яa-я]+)")
    cities = cities.iloc[:, 0].fillna(cities.iloc[:, 1])
    cities_translation = {}
    for city in cities.unique():
        cities_translation[city] = translate.ru_to_en(city)
    shops["city"] = cities.map(cities_translation)
    shops.loc[shops["city"] == "Czechs", "city"] = "Chekhov"
    shops.loc[shops["city"] == "the Internet", "city"] = "Online"
    shops.loc[shops["city"] == "exit", "city"] = "Door-to-door sales"
    shops.loc[shops["city"] == "SPb", "city"] = "St. Petersburg"

    shops["shop_type"] = shops["shop_name"].str.extract("([A-Я]{2,})")
    shops["shop_subname"] = shops["shop_name"].str.extract('"(.*?)"')

    return shops[["shop_id", "shop_name", "city", "shop_type", "shop_subname"]]


@reindex
def extract_item_category_features(item_categories):
    item_categories["item_supcategory_name"] = item_categories["item_category_name"].str.extract("(^[^-]*[^ -])")
    item_categories.loc[item_categories["item_supcategory_name"] == "Карты оплаты (Кино, Музыка, Игры)",
                        "item_supcategory_name"] = "Карты оплаты"
    item_categories.loc[item_categories["item_supcategory_name"] == "Чистые носители (шпиль)",
                        "item_supcategory_name"] = "Чистые носители"
    item_categories.loc[item_categories["item_supcategory_name"] == "Чистые носители (штучные)",
                        "item_supcategory_name"] = "Чистые носители"

    item_categories["item_subcategory_name"] = item_categories["item_category_name"].str.extract("([^ -][^-]*$)")

    return item_categories[["item_category_id", "item_category_name", "item_subcategory_name", "item_supcategory_name"]]


def filter_list_column(s, n):
    c = Counter([item for lst in s.tolist() for item in lst])
    to_include = [item for item in c if c[item] > n]
    try:
        to_include.remove(None)
    except ValueError:
        pass
    return s.apply(lambda lst: [el for el in lst if el in to_include])


@reindex
def extract_item_features(items):

    items["item_label1"] = items["item_name"] \
        .str.lower() \
        .str.extractall(r"\[(.*?)\]") \
        [0] \
        .str.split(",") \
        .groupby(level=0) \
        .apply(lambda l: [item.strip() for sublist in l for item in sublist])
    items["item_label1"] = items["item_label1"] \
        .where(items["item_label1"].notnull(), None) \
        .apply(lambda l: l if type(l) is list else [l])

    items["item_label2"] = items["item_name"] \
        .str.lower() \
        .str.extractall(r"\((.*?)\)") \
        [0] \
        .str.split(",") \
        .groupby(level=0) \
        .apply(lambda l: [item.strip() for sublist in l for item in sublist])
    items["item_label2"] = items["item_label2"] \
        .where(items["item_label2"].notnull(), None) \
        .apply(lambda l: l if type(l) is list else [l])

    items["item_subname"] = items["item_name"].str.lower().str.extract(r"(^[^\(\[]*[^ \(\[])")

    items["item_label1"] = filter_list_column(items["item_label1"], 200)
    mlb = MultiLabelBinarizer()
    item_label1_encoded = pd.DataFrame(mlb.fit_transform(items["item_label1"]), columns=mlb.classes_)
    item_label1_encoded.columns = "l1_" + item_label1_encoded.columns
    items["item_label2"] = filter_list_column(items["item_label2"], 200)
    mlb = MultiLabelBinarizer()
    item_label2_encoded = pd.DataFrame(mlb.fit_transform(items["item_label2"]), columns=mlb.classes_)
    item_label2_encoded.columns = "l2_" + item_label2_encoded.columns
    items = pd.concat([items[["item_id"]], item_label1_encoded, item_label2_encoded, items[["item_subname"]]], axis=1)

    return items


@reindex
def extract_calendar_features(calendar):

    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["month_index"] = calendar["date"].dt.to_period("M").dt.to_timestamp()
    calendar["non_workday"] = (calendar["weekday"].isin([6, 7]) | calendar["bank_holiday"].notnull()).astype(int)

    calendar_features = calendar.groupby("month_index").agg({"date": len,
                                                             "non_workday": sum,
                                                             "bank_holiday": lambda x: x.notnull().sum(),
                                                             "weekday": [lambda x: (x == 1).sum(),
                                                                         lambda x: (x == 2).sum(),
                                                                         lambda x: (x == 3).sum(),
                                                                         lambda x: (x == 4).sum(),
                                                                         lambda x: (x == 5).sum(),
                                                                         lambda x: (x == 6).sum(),
                                                                         lambda x: (x == 7).sum()]})
    calendar_features.columns = ["n_days", "n_nonworking_days", "n_bank_holidays",
                                 "n_mondays", "n_tuesdays", "n_wednesdays", "n_thursdays", "n_fridays",
                                 "n_saturdays", "n_sundays"]
    calendar_features.reset_index(inplace=True)
    calendar_features["month"] = calendar_features["month_index"].dt.month
    calendar_features["year"] = calendar_features["month_index"].dt.year

    return calendar_features.rename({"month_index": "date"}, axis=1)


@reindex
def extract_possible_discount(sales_train):
    """Should be applied to original (daily) data frame."""
    df = sales_train.copy()
    df["possible_discount"] = \
        (~np.isclose(df["item_price"], df["item_price"].round(2), rtol=1e-8)).astype(int)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    res_df = df \
        .groupby(["date", "date_block_num", "shop_id", "item_id"])["possible_discount"] \
        .agg([sum,
              lambda x: x.sum() / x.size])
    res_df.columns = ["possible_discounts_n", "possible_discounts_prop"]
    return res_df.reset_index()


@reindex
def extract_lagged_features(df, agg_mapping, by, lags=[1, 2, 3, 12], fill_value=None):
    res_df = pd.DataFrame()
    if not isinstance(by, list):
        by_str = by
        by = [by]
    else:
        by_str = "_".join(by)
    by.insert(0, "date")
    aggregated_df = df.groupby(by).agg(agg_mapping)
    by.remove("date")
    for c in agg_mapping.keys():
        for l in lags:
            res_df["_".join([c, "by", by_str, "lag", str(l)])] = aggregated_df \
                .sort_values("date") \
                .groupby(by)[c] \
                .shift(l, fill_value=fill_value)

    return res_df.reset_index()


@reindex
def extract_missingness_patterns(df, by, index):
    if isinstance(by, list):
        by_str = "_".join(by)
    else:
        by_str = by
        by = [by]
        # TODO: Should we replace negative scores with np.nan?

    if not isinstance(index, list):
        index = [index]

    if len(index) > 1:
        index = pd.MultiIndex.from_product(list([*index]))
        index = [index]

    dfs = list()
    for train_id, val_id in tqdm(GroupTimeSeriesSplit(n_splits=df["date_block_num"].nunique() - 1) \
                                         .split(df, y=None, groups=df["date_block_num"])):
        current_month = df.loc[val_id, "date_block_num"].unique()[0]
        current_df = df \
            .loc[train_id] \
            .pivot_table(values="item_cnt_month", index=by,
                         columns="date_block_num", aggfunc="sum") \
            .reindex(*index) \
            .apply([lambda row: row.first_valid_index(),
                    lambda row: current_month - row.last_valid_index() - 1,
                    lambda row: row.isnull().mean()], axis=1)
        dfs.append(current_df)

    res_df = pd.concat(dfs, axis=0,
                       keys=df["date_block_num"].unique(),
                       names=["date_block_num", *by])
    res_df.columns.name = None
    res_df.rename({"<lambda_0>": "first_nonmissing_month_by_" + by_str,
                   "<lambda_1>": "n_months_since_last_nonmissing_" + by_str,
                   "<lambda_2>": "prop_missing_months_by_" + by_str}, axis=1, inplace=True)
    return res_df.reset_index()


if __name__ == "__main__":

    sales_by_month, sales_train, test, items, item_categories, shops, calendar = read_preprocessed_data()

    item_features = extract_item_features(sales_by_month, items, on="item_id")
    print("Item features extracted.")
    shop_features = extract_shop_features(sales_by_month, shops, on="shop_id")
    print("Shop features extracted.")

    item_category_features = extract_item_category_features(sales_by_month, item_categories, on="item_category_id")
    print("Item category features extracted.")

    calendar_features = extract_calendar_features(sales_by_month, calendar, on="date")
    print("Calendar features extracted.")

    # TODO: Check why there are so many NaNs in discount_features
    discount_features = extract_possible_discount(sales_by_month, sales_train,
                                                  on=["date", "date_block_num", "shop_id", "item_id"])
    print("Discount features extracted.")

    lagged_features_by_shop_id = extract_lagged_features(main_df=sales_by_month,
                                                         mapping_df=sales_by_month,
                                                         on=["date", "shop_id"],
                                                         columns=["median_item_price", "item_cnt_month"],
                                                         by="shop_id",
                                                         lags=[1, 2, 3, 12],
                                                         fill_value=None)
    lagged_features_by_item_id = extract_lagged_features(main_df=sales_by_month,
                                                         mapping_df=sales_by_month,
                                                         on=["date", "item_id"],
                                                         columns=["median_item_price", "item_cnt_month"],
                                                         by="item_id",
                                                         lags=[1, 2, 3, 12],
                                                         fill_value=None)
    lagged_features_by_item_category_id = extract_lagged_features(main_df=sales_by_month,
                                                                  mapping_df=sales_by_month,
                                                                  on=["date", "item_category_id"],
                                                                  columns=["median_item_price", "item_cnt_month"],
                                                                  by="item_category_id",
                                                                  lags=[1, 2, 3, 12],
                                                                  fill_value=None)
    lagged_features_by_shop_id_item_id = extract_lagged_features(main_df=sales_by_month,
                                                                 mapping_df=sales_by_month,
                                                                 on=["date", "shop_id", "item_id"],
                                                                 columns=["median_item_price", "item_cnt_month"],
                                                                 by=["shop_id", "item_id"],
                                                                 lags=[1, 2, 3, 12],
                                                                 fill_value=None)
    lagged_features_by_shop_id_item_category_id = extract_lagged_features(main_df=sales_by_month,
                                                                          mapping_df=sales_by_month,
                                                                          on=["date", "shop_id", "item_category_id"],
                                                                          columns=["median_item_price", "item_cnt_month"],
                                                                          by=["shop_id", "item_category_id"],
                                                                          lags=[1, 2, 3, 12],
                                                                          fill_value=None)
    print("Lagged features extracted.")

    missingness_features_by_shop_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                   mapping_df=sales_by_month,
                                                                   on=["date_block_num", "shop_id"],
                                                                   by="shop_id",
                                                                   index=shops["shop_id"])
    missingness_features_by_item_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                   mapping_df=sales_by_month,
                                                                   on=["date_block_num", "item_id"],
                                                                   by="item_id",
                                                                   index=items["item_id"])
    missingness_features_by_item_category_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                            mapping_df=sales_by_month,
                                                                            on=["date_block_num", "item_category_id"],
                                                                            by="item_category_id",
                                                                            index=item_categories["item_category_id"])
    missingness_features_by_shop_id_item_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                           mapping_df=sales_by_month,
                                                                           on=["date_block_num", "shop_id", "item_id"],
                                                                           by=["shop_id", "item_id"],
                                                                           index=[shops["shop_id"], items["item_id"]])
    missingness_features_by_shop_id_item_category_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                                    mapping_df=sales_by_month,
                                                                                    on=["date_block_num", "shop_id",
                                                                                        "item_category_id"],
                                                                                    by=["shop_id", "item_category_id"],
                                                                                    index=[shops["shop_id"],
                                                                                           item_categories["item_category_id"]])
    print("Missingness features extracted.")

