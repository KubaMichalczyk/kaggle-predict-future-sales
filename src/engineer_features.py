import pandas as pd
import numpy as np
import translate
import argparse
from preprocess_data import read_preprocessed_data
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from utils import custom_skipna, GroupTimeSeriesSplit
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
    shops.loc[shops["city"] == "Outgoing", "city"] = "Door-to-door sales"
    shops.loc[shops["city"] == "SPb", "city"] = "St. Petersburg"

    shops["shop_type"] = shops["shop_name"].str.extract("([A-Я]{2,})")
    shops["shop_subname"] = shops["shop_name"].str.extract('"(.*?)"')

    return shops[["shop_id", "city", "shop_type", "shop_subname"]]


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

    return item_categories[["item_category_id", "item_subcategory_name", "item_supcategory_name"]]


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
def extract_lagged_features(df, agg_mapping, by, lags=[1, 2, 3, 12], fill_value=None):
    if not isinstance(by, list):
        by_str = by
        by = [by]
    else:
        by_str = "_".join(by)
    by.insert(0, "date_block_num")
    aggregated_df = df.groupby(by).agg(agg_mapping).reset_index()
    if type(aggregated_df.columns) == pd.core.indexes.multi.MultiIndex:
        aggregated_df.columns = ['_'.join(col).strip("_") for col in aggregated_df.columns.values]
    by.remove("date_block_num")
    res_df = aggregated_df[["date_block_num"] + by]
    for k in agg_mapping.keys():
        for c in aggregated_df.columns[aggregated_df.columns.str.startswith(k)]:
            for l in lags:
                shifted = aggregated_df[["date_block_num"] + by + [c]].copy()
                shifted.loc[:, "date_block_num"] += l
                shifted.rename({c: "_".join([c, "by", by_str, "lag", str(l)])}, axis=1, inplace=True)
                res_df = res_df.merge(shifted, how="left", on=["date_block_num"] + by)
    return res_df


@reindex
def extract_rolling_window_features(df, window, agg_func, by):
    if not isinstance(by, list):
        by_str = by
        by = [by]
    else:
        by_str = "_".join(by)
    by.insert(0, "date_block_num")
    aggregated_df = df.groupby(by)[["item_cnt_month"]].sum().reset_index()
    by.remove("date_block_num")
    res_df = aggregated_df[["date_block_num"] + by]
    
    if not isinstance(window, list):
        window=[window]
    for w in window:
        rolling_stats = aggregated_df.groupby(by)["item_cnt_month"].rolling(w).agg(agg_func)
        rolling_stats.columns = "item_cnt_month_rolling_" + rolling_stats.columns + "_of_" + str(w) + "_by_" + by_str
        shifted = pd.concat([res_df[["date_block_num"] + by], rolling_stats.reset_index(by, drop=True)], 
                            axis=1)
        shifted["date_block_num"] += 1
        res_df = res_df.merge(shifted, how="left", on=["date_block_num"] + by)
    return res_df


@reindex
def extract_missingness_patterns(df, by, index):
    if isinstance(by, list):
        by_str = "_".join(by)
    else:
        by_str = by
        by = [by]

    if not isinstance(index, list):
        index = [index]

    df.loc[df["item_cnt_month"] <= 0, "item_cnt_month"] = np.nan
    dfs = list()
    for train_id, val_id in tqdm(GroupTimeSeriesSplit(n_splits=df["date_block_num"].nunique() - 1)
                                 .split(df, y=None, groups=df["date_block_num"])):
        current_month = df.loc[val_id, "date_block_num"].unique()[0]
        new_index = pd.MultiIndex.from_product(
            list([*index, pd.Series(np.arange(0, current_month), name="date_block_num")]))
        df_train = df.loc[train_id]
        prop_nonmissing = df_train \
            .groupby(by + ["date_block_num"])["item_cnt_month"] \
            .sum(min_count=1) \
            .reindex(new_index) \
            .isnull() \
            .groupby(by) \
            .mean()
        last_nonmissing = df_train.groupby(by)["date_block_num"].max()
        first_nonmissing = df_train.groupby(by)['date_block_num'].min()
        dfs.append(
            pd.concat([prop_nonmissing, last_nonmissing, first_nonmissing], axis=1))

    res_df = pd.concat(dfs, axis=0,
                       keys=df["date_block_num"].unique() + 1,
                       names=["date_block_num", *by])
    res_df.columns.name = None
    res_df.columns = ["prop_missing_months_by_" + by_str,
                      "last_nonmissing_month_by_" + by_str,
                      "first_nonmissing_month_by_" + by_str]
    res_df = res_df.astype(np.float32)
    res_df.reset_index(inplace=True)
    res_df["n_months_since_first_nonmissing_by_" + by_str] = res_df["date_block_num"] - \
        res_df["first_nonmissing_month_by_" + by_str]
    res_df["n_months_since_last_nonmissing_by_" + by_str] = res_df["date_block_num"] - \
        res_df["last_nonmissing_month_by_" + by_str]
    return res_df


@reindex
def extract_item_embeddings(items, embed_model):
    item_embeddings = pd.DataFrame(embed_model["item_embeddings.weight"].cpu().detach().numpy())
    item_embeddings.columns = ["item_e" + str(i) for i in item_embeddings.columns]
    items_with_embeddings = pd.concat([items,
                                       item_embeddings],
                                      axis=1)
    return items_with_embeddings


@reindex
def extract_item_category_embeddings(item_categories, embed_model):
    item_category_embeddings = pd.DataFrame(embed_model["item_category_embeddings.weight"].cpu().detach().numpy())
    item_category_embeddings.columns = ["item_category_e" + str(i) for i in item_category_embeddings.columns]
    item_categories_with_embeddings = pd.concat([item_categories,
                                                 item_category_embeddings],
                                                axis=1)
    return item_categories_with_embeddings


@reindex
def extract_shop_embeddings(shops, embed_model):
    shop_embeddings = pd.DataFrame(embed_model["shop_embeddings.weight"].cpu().detach().numpy())
    shop_embeddings.columns = ["shop_e" + str(i) for i in shop_embeddings.columns]
    shops_with_embeddings = pd.concat([shops,
                                       shop_embeddings],
                                      axis=1)
    return shops_with_embeddings


def drop_duplicated_features(df):
    """Drops all duplicated features from the data frame."""

    duplicated_cols = set()
    for i in tqdm(range(df.shape[1])):
        c1 = df.iloc[:, i]
        for j in range(i + 1, df.shape[1]):
            c2 = df.iloc[:, j]
            if c1.equals(c2):
                duplicated_cols.add(df.columns[j])
    return df.drop(columns=duplicated_cols)


def custom_sum(x):
    """Custom wrapper to preserve function name in aggregates."""
    return custom_skipna(x, np.sum)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract the features.")
    parser.add_argument("-i", "--disable_item_features",
                        dest="item_features",  action="store_false",
                        help='Whether to disable extracting item features.')
    parser.add_argument("-s", "--disable_shop_features",
                        dest="shop_features",  action="store_false",
                        help='Whether to disable extracting shop features.')
    parser.add_argument("-ic", "--disable_item_category_features",
                        dest="item_category_features",  action="store_false",
                        help='Whether to disable extracting item category features.')
    parser.add_argument("-c", "--disable_calendar_features",
                        dest="calendar_features",  action="store_false",
                        help='Whether to disable extracting calendar features.')
    parser.add_argument("-l", "--disable_lagged_features",
                        dest="lagged_features",  action="store_false",
                        help='Whether to disable extracting lagged features.')
    parser.add_argument("-r", "--disable_rolling_features",
                        dest="rolling_features",  action="store_false",
                        help='Whether to disable extracting rolling window features.')
    parser.add_argument("-m", "--disable_missingness_features",
                        dest="missingness_features",  action="store_false",
                        help='Whether to disable extracting missingness features.')
    parser.add_argument("-e", "--disable_embedding_features",
                        dest="embedding_features",  action="store_false",
                        help='Whether to disable extracting embedding features.')
    parser.add_argument("-dd", "--drop_duplicated_features",
                        dest="drop_duplicated_features",  action="store_true",
                        help='Whether to drop duplicated features (if there are any).')
    args = parser.parse_args()

    sales_by_month, sales_train, test, items, item_categories, shops, calendar = read_preprocessed_data()
    
    all_features = [sales_by_month[['date_block_num',
                                   'ID',
                                   'shop_id',
                                   'item_category_id',
                                   'item_id', 
                                   'item_cnt_month']]]

    if args.item_features:
        item_features = extract_item_features(sales_by_month, items, on="item_id")
        all_features.append(item_features)
        print("Item features extracted.")

    if args.shop_features:
        shop_features = extract_shop_features(sales_by_month, shops, on="shop_id")
        all_features.append(shop_features)
        print("Shop features extracted.")

    if args.item_category_features:
        item_category_features = extract_item_category_features(sales_by_month, item_categories, on="item_category_id")
        all_features.append(item_category_features)
        print("Item category features extracted.")

    if args.calendar_features:
        calendar_features = extract_calendar_features(sales_by_month, calendar, on="date")
        all_features.append(calendar_features)
        print("Calendar features extracted.")

    if args.lagged_features:
        
        lagged_features_aggregate = extract_lagged_features(main_df=sales_by_month,
                                                            mapping_df=sales_by_month,
                                                            on=["date_block_num"],
                                                            agg_mapping={"item_cnt_month":
                                                                        [custom_sum,
                                                                        np.nanmean],
                                                                        "mean_item_price": np.nanmean,
                                                                        "median_item_price": np.nanmedian,
                                                                        "mean_item_revenue": np.nanmean,
                                                                        "median_item_revenue": np.nanmedian,
                                                                        "possible_discounts_n": custom_sum,
                                                                        "possible_discounts_prop": np.nanmean},
                                                            by=[],
                                                            lags=[1, 2, 3, 12],
                                                            fill_value=None)
        all_features.append(lagged_features_aggregate)
        
        lagged_features_by_shop_id = extract_lagged_features(main_df=sales_by_month,
                                                             mapping_df=sales_by_month,
                                                             on=["date_block_num",
                                                                 "shop_id"],
                                                             agg_mapping={"item_cnt_month":
                                                                          [custom_sum, 
                                                                          np.nanmean],
                                                                          "mean_item_price": np.nanmean,
                                                                          "median_item_price": np.nanmedian,
                                                                          "mean_item_revenue": np.nanmean,
                                                                          "median_item_revenue": np.nanmedian,
                                                                          "possible_discounts_n": custom_sum,
                                                                          "possible_discounts_prop": np.nanmean},
                                                             by="shop_id",
                                                             lags=[1, 2, 3, 12],
                                                             fill_value=None)
        all_features.append(lagged_features_by_shop_id)

        lagged_features_by_item_id = extract_lagged_features(main_df=sales_by_month,
                                                             mapping_df=sales_by_month,
                                                             on=["date_block_num", 
                                                                 "item_id"],
                                                             agg_mapping={"item_cnt_month":
                                                                          [custom_sum, 
                                                                          np.nanmean],
                                                                          "mean_item_price": np.nanmean,
                                                                          "median_item_price": np.nanmedian,
                                                                          "mean_item_revenue": np.nanmean,
                                                                          "median_item_revenue": np.nanmedian,
                                                                          "possible_discounts_n": custom_sum,
                                                                          "possible_discounts_prop": np.nanmean},
                                                             by="item_id",
                                                             lags=[1, 2, 3, 12],
                                                             fill_value=None)
        all_features.append(lagged_features_by_item_id)
        
        lagged_features_by_item_category_id = extract_lagged_features(main_df=sales_by_month,
                                                                      mapping_df=sales_by_month,
                                                                      on=["date_block_num",
                                                                          "item_category_id"],
                                                                      agg_mapping={"item_cnt_month":
                                                                                   [custom_sum,
                                                                                    np.nanmean],
                                                                                   "mean_item_price": np.nanmean,
                                                                                   "median_item_price": np.nanmedian,
                                                                                   "mean_item_revenue": np.nanmean,
                                                                                   "median_item_revenue": np.nanmedian,
                                                                                   "possible_discounts_n": custom_sum,
                                                                                   "possible_discounts_prop": np.nanmean},
                                                                      by="item_category_id",
                                                                      lags=[1, 2, 3, 12],
                                                                      fill_value=None)
        all_features.append(lagged_features_by_item_category_id)

        lagged_features_by_shop_id_item_id = extract_lagged_features(main_df=sales_by_month,
                                                                     mapping_df=sales_by_month,
                                                                     on=["date_block_num",
                                                                         "shop_id", 
                                                                         "item_id"],
                                                                     agg_mapping={"item_cnt_month":
                                                                                   [custom_sum,
                                                                                    np.nanmean],
                                                                                  "mean_item_price": np.nanmean,
                                                                                  "median_item_price": np.nanmedian,
                                                                                  "mean_item_revenue": np.nanmean,
                                                                                  "median_item_revenue": np.nanmedian,
                                                                                  "possible_discounts_n": custom_sum,
                                                                                  "possible_discounts_prop": np.nanmean},
                                                                     by=["shop_id",
                                                                         "item_id"],
                                                                     lags=[1, 2, 3, 12],
                                                                     fill_value=None)        
        all_features.append(lagged_features_by_shop_id_item_id)

        lagged_features_by_shop_id_item_category_id = extract_lagged_features(main_df=sales_by_month,
                                                                              mapping_df=sales_by_month,
                                                                              on=["date_block_num",
                                                                                  "shop_id", 
                                                                                  "item_category_id"],
                                                                              agg_mapping={"item_cnt_month":
                                                                                           [custom_sum,
                                                                                            np.nanmean],
                                                                                           "mean_item_price":
                                                                                           np.nanmean,
                                                                                           "median_item_price": 
                                                                                           np.nanmedian,
                                                                                           "mean_item_revenue": 
                                                                                           np.nanmean,
                                                                                           "median_item_revenue": 
                                                                                           np.nanmedian,
                                                                                           "possible_discounts_n":
                                                                                           custom_sum,
                                                                                           "possible_discounts_prop": 
                                                                                           np.nanmean},
                                                                              by=["shop_id",
                                                                                  "item_category_id"],
                                                                              lags=[1, 2, 3, 12],
                                                                              fill_value=None)
        all_features.append(lagged_features_by_shop_id_item_category_id)
        
        print("Lagged features extracted.")

    if args.rolling_features:

        rolling_features_by_shop_id = extract_rolling_window_features(main_df=sales_by_month, 
                                                                      mapping_df=sales_by_month, 
                                                                      on=["date_block_num", "shop_id"],
                                                                      window=[3, 6, 12], 
                                                                      agg_func=[np.nanmean, np.nanstd], 
                                                                      by="shop_id")
        all_features.append(rolling_features_by_shop_id)

        rolling_features_by_item_id = extract_rolling_window_features(main_df=sales_by_month, 
                                                                      mapping_df=sales_by_month, 
                                                                      on=["date_block_num", "item_id"],
                                                                      window=[3, 6, 12], 
                                                                      agg_func=[np.nanmean, np.nanstd], 
                                                                      by="item_id")
        all_features.append(rolling_features_by_item_id)

        rolling_features_by_item_category_id = extract_rolling_window_features(main_df=sales_by_month,
                                                                               mapping_df=sales_by_month,
                                                                               on=["date_block_num", "item_category_id"],
                                                                               window=[3, 6, 12],
                                                                               agg_func=[np.nanmean, np.nanstd],
                                                                               by="item_category_id")
        all_features.append(rolling_features_by_item_category_id)

        rolling_features_by_shop_id_item_id = extract_rolling_window_features(main_df=sales_by_month, 
                                                                              mapping_df=sales_by_month, 
                                                                              on=["date_block_num", "shop_id", "item_id"],
                                                                              window=[3, 6, 12], 
                                                                              agg_func=[np.nanmean, np.nanstd], 
                                                                              by=["shop_id", "item_id"])
        all_features.append(rolling_features_by_shop_id_item_id)
        
        rolling_features_by_shop_id_item_category_id = extract_rolling_window_features(main_df=sales_by_month, 
                                                                                       mapping_df=sales_by_month,
                                                                                       on=["date_block_num", 
                                                                                           "shop_id", 
                                                                                           "item_category_id"],
                                                                                       window=[3, 6, 12],
                                                                                       agg_func=[np.nanmean, np.nanstd],
                                                                                       by=["shop_id", 
                                                                                           "item_category_id"])
        all_features.append(rolling_features_by_shop_id_item_category_id)

        print("Rolling window features extracted.")

    if args.missingness_features:

        missingness_features_by_shop_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                       mapping_df=sales_by_month,
                                                                       on=["date_block_num", "shop_id"],
                                                                       by="shop_id",
                                                                       index=shops["shop_id"])
        all_features.append(missingness_features_by_shop_id)

        missingness_features_by_item_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                       mapping_df=sales_by_month,
                                                                       on=["date_block_num",
                                                                           "item_id"],
                                                                       by="item_id",
                                                                       index=items["item_id"])
        all_features.append(missingness_features_by_item_id)

        missingness_features_by_item_category_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                                mapping_df=sales_by_month,
                                                                                on=["date_block_num",
                                                                                    "item_category_id"],
                                                                                by="item_category_id",
                                                                                index=item_categories["item_category_id"])
        all_features.append(missingness_features_by_item_category_id)

        missingness_features_by_shop_id_item_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                               mapping_df=sales_by_month,
                                                                               on=["date_block_num",
                                                                                   "shop_id", "item_id"],
                                                                               by=["shop_id",
                                                                                   "item_id"],
                                                                               index=[shops["shop_id"], items["item_id"]])
        all_features.append(missingness_features_by_shop_id_item_id)

        missingness_features_by_shop_id_item_category_id = extract_missingness_patterns(main_df=sales_by_month,
                                                                                        mapping_df=sales_by_month,
                                                                                        on=["date_block_num", "shop_id",
                                                                                            "item_category_id"],
                                                                                        by=["shop_id", "item_category_id"],
                                                                                        index=[shops["shop_id"],
                                                                                               item_categories[
                                                                                                   "item_category_id"]])
        all_features.append(missingness_features_by_shop_id_item_category_id)

        print("Missingness features extracted.")

    if args.embedding_features:
        
        embed_model = torch.load("../models/embedding_model.pt")
        
        item_embeddings = extract_item_embeddings(sales_by_month,
                                                  items[["item_id"]],
                                                  on=["item_id"],
                                                  embed_model=embed_model)
        all_features.append(item_embeddings)


        item_category_embeddings = extract_item_category_embeddings(sales_by_month,
                                                                    item_categories[["item_category_id"]], 
                                                                    on=["item_category_id"], 
                                                                    embed_model=embed_model)
        all_features.append(item_category_embeddings)


        shop_embeddings = extract_shop_embeddings(sales_by_month, 
                                                  shops[["shop_id"]], 
                                                  on="shop_id", 
                                                  embed_model=embed_model)
        all_features.append(shop_embeddings)

        print("Embedding features extracted.")


    all_features = pd.concat(all_features, axis=1)
    if args.drop_duplicated_features:
        all_features = drop_duplicated_features(all_features)
    
    all_features.to_parquet("../input/all_features.parquet")
