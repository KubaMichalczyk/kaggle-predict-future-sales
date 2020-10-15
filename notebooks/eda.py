# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from time import sleep

import sys
sys.path.insert(1, "/media/data/kaggle/predict-future-sales")
from src import translate

# %matplotlib inline
# -

# %run -i "../src/read_data.py"

sales_train.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

# **item_price** - big standard deviation, negative values (`-1`) and some outliers (e.g. 307980)
#
# **item_cnt_day** - big standard deviation (> twice the mean), negative values (e.g. -22) and some outliers (e.g 2169)

sales_train.isnull().sum(axis=0)

sales_train.sort_values(["date", "date_block_num", "shop_id", "item_id"], inplace=True)

sales_train = sales_train \
    .merge(items, on="item_id", how="left") \
    .merge(item_categories, on="item_category_id", how="left") \
    .merge(shops, on="shop_id", how = "left")

sales_train_by_month = sales_train \
    .groupby(["shop_id", "item_id", sales_train["date"].dt.to_period("M").dt.to_timestamp()]) \
    [["item_price", "item_cnt_day"]] \
    .agg({"item_price": [np.mean, np.median], "item_cnt_day": np.sum}) \
    .reset_index()
sales_train_by_month.columns = ["_".join(col[::-1]).strip("_") for col in sales_train_by_month.columns]
sales_train_by_month.rename({"sum_item_cnt_day": "item_cnt_month"}, axis=1, inplace=True)

sales_train_by_month = sales_train_by_month \
    .merge(items, on="item_id", how="left") \
    .merge(item_categories, on="item_category_id", how="left") \
    .merge(shops, on="shop_id", how = "left")

# # Explore shops information

shops

# `shop_name` seems to be structured so that the City appears first, then is followed by probably some shop type marked with capital letters (TC, SEC, TSC, TK) and finaly the name in the quotation marks.

shops["shop_name"].value_counts()

# Names are unique, but we can immediately notice that the shops no. 10 and no. 11 are probably the same shop. We'll try to extract features according to the naming structure and then try to find another possible duplicates. We'll do that using original Russian names and translate to English later, as some information might have been lost in translation process.

cities = shops["shop_name"].str.extract("([A-Яa-я]+\\.[A-Яa-я]+)|([A-Яa-я]+)")

cities = cities.iloc[:, 0].fillna(cities.iloc[:, 1])

cities_translation = {}
for city in cities.unique():
    cities_translation[city] = translate.ru_to_en(city)
    sleep(30)

shops["city"] = cities.map(cities_translation)

shops

shops.loc[shops["city"].isin(["Digital", "the Internet", "exit", "Czechs", "SPb"])]

shops.loc[shops["city"] == "Czechs", "city"] = "Chekhov"
shops.loc[shops["city"] == "the Internet", "city"] = "Online"
shops.loc[shops["city"] == "exit", "city"] = "Door-to-door sales"
shops.loc[shops["city"] == "SPb", "city"] = "St. Petersburg"

shops

shops["shop_type"] = shops["shop_name"].str.extract("([A-Я]{2,})")

shops["shop_type"].unique()

shop_types_translation = {}
for shop_type in shops["shop_type"].unique():
    shop_types_translation[city] = translate.ru_to_en(shop_type)
    sleep(30)

shops["name_only"] = shops["shop_name"].str.extract('"(.*?)"')

possible_duplicates = shops[["city", "shop_type", "name_only"]].fillna("None").groupby(["city", "shop_type", "name_only"]).size().sort_values(ascending=False)
possible_duplicates[possible_duplicates > 1]

# There are 5 possible duplicates. 

shops_text_nans = shops.fillna("None")
shops.loc[(shops["city"] == "Zhukovsky") & (shops_text_nans["shop_type"] == "None") & (shops_text_nans["name_only"] == "None")]

sales_train_by_month.loc[sales_train_by_month["shop_id"].isin([10, 11])].groupby("date").apply(lambda x: x["shop_id"].unique())

# Shops no. 10 and no. 11 are definitely the same shop, with just the typo in a name in one month.

shops.loc[(shops["city"] == "Yakutsk") & (shops["shop_type"] == "ТЦ") & (shops["name_only"] == "Центральный")]

sales_train_by_month.loc[sales_train_by_month["shop_id"].isin([1, 58])].groupby("date").apply(lambda x: x["shop_id"].unique())

# Shop no. 1 probably became shop no. 58 after two months. The name is almost the same; I'm not sure what "Franc" at the end of original name means (maybe an abbreviation from "franchise"), but I believe there is a high chance that this is exactly the same shop, or with some slight change.

shops.loc[(shops["city"] == "Yakutsk") & (shops_text_nans["shop_type"] == "None") & (shops_text_nans["name_only"] == "None")]

sales_train_by_month.loc[sales_train_by_month["shop_id"].isin([0, 57])].groupby("date").apply(lambda x: x["shop_id"].unique())

# The same as above, I'll assume that this is the same shop.

shops.loc[(shops["city"] == "Rostov-on-Don") & (shops["shop_type"] == "ТРК") & (shops["name_only"] == "Мегацентр Горизонт")]

sales_train_by_month.loc[sales_train_by_month["shop_id"].isin([39, 40])].groupby("date").apply(lambda x: x["shop_id"].unique())

shops.loc[(shops["city"] == "Moscow") & (shops["shop_type"] == "ТК") & (shops["name_only"] == "Буденовский")]

sales_train_by_month.loc[sales_train_by_month["shop_id"].isin([23, 24])].groupby("date").apply(lambda x: x["shop_id"].unique())

# With shops no. 39 and 40 or no. 23 and 24 the situation is different, those shops were probably different POS in the same location.

# We create a new `shop_id`, leaving the old one as it is - we'll check later which way gives the better model performance.

shops["shop_id_new"] = shops["shop_id"]
shops.loc[shops["shop_id"] == 11, "shop_id_new"] = 10
shops.loc[shops["shop_id"] == 57, "shop_id_new"] = 0
shops.loc[shops["shop_id"] == 58, "shop_id_new"] = 1

# # Explore item categories

pd.options.display.max_rows=None

item_categories

# There are two categories for tickets (no. 8 and 80), probably one for digital tickets and one for standard paper tickets, but let's make sure that the one didn't replace another at some point.

sales_train_by_month.loc[sales_train_by_month["item_category_id"].isin([8, 80])].groupby("date").apply(lambda x: x["item_category_id"].unique())

# No, it's not the case - they're certainly two separate, although similar categories.

item_categories["item_supcategory_name"] = item_categories["item_category_name"].str.extract("(^[^-]*[^ -])")

item_categories

item_categories.loc[item_categories["item_supcategory_name"] == "Карты оплаты (Кино, Музыка, Игры)", "item_supcategory_name"] = "Карты оплаты"
item_categories.loc[item_categories["item_supcategory_name"] == "Чистые носители (шпиль)", "item_supcategory_name"] = "Чистые носители"
item_categories.loc[item_categories["item_supcategory_name"] == "Чистые носители (штучные)", "item_supcategory_name"] = "Чистые носители"

item_categories["item_subcategory_name"] = item_categories["item_category_name"].str.extract("([^ -][^-]*$)")

item_categories

pd.options.display.max_rows = 60

# # Explore items

items.sample(frac=1).head(60)

# Apart from the item name, there are some additional features stored in square and round brackets (e.g. game platform or language etc.). The name structure indicates that those features are somehow related (i.e. the characteristics put in square brackets are some categories). Also, we often have many characteristics on the same level (either two square brackets or multiple values separated by a comma within the same square bracket).

items["item_feature1"] = items["item_name"] \
    .str.lower() \
    .str.extractall("\[(.*?)\]") \
    [0] \
    .str.split(",") \
    .groupby(level=0) \
    .apply(lambda l: [item.strip() for sublist in l for item in sublist])

items["item_feature1"].explode().value_counts()

items["item_feature2"] = items["item_name"] \
    .str.lower() \
    .str.extractall("\((.*?)\)") \
    [0] \
    .str.split(",") \
    .groupby(level=0) \
    .apply(lambda l: [item.strip() for sublist in l for item in sublist])

items["item_feature2"].explode().value_counts()

items["item_feature3"] = items["item_name"].str.lower().str.extract("(^[^\(\[]*[^ \(\[])")

items["item_feature3"].value_counts(dropna=False).head(20)

# Low count feature - won't be probably useful in this form.

# Generally, it feels that much more can be extracted from `item_name`. Maybe some pre-trained embeddings would be worth to try later.

# # Explore the training set and test set

sales_train_by_month = sales_train_by_month.merge(test, how="left", on=["shop_id", "item_id"])

(sales_train_by_month.loc[sales_train_by_month["ID"].isnull(), ["shop_id", "item_id"]].drop_duplicates().shape[0] /
 sales_train_by_month[["shop_id", "item_id"]].drop_duplicates().shape[0])

# Almost 74% of (`shop_id`, `item_id`) combinations present in training data are absent in test data. However, these may still be useful for the model.

train_pairs = sales_train[["shop_id", "item_id"]].drop_duplicates()
train_pairs["set"] = "train"

test.merge(train_pairs, how="left", on=["shop_id", "item_id"])["set"].value_counts(dropna=False, normalize=True)

# About 48% of pairs from the test set doesn't appear in the training set! We'll have a hard task with predicting for them, but we can probably lean back a little on global statistics for either `shop_id` or `item_id`

len(set(test["shop_id"]).difference(set(sales_train["shop_id"])))

len(set(test["item_id"]).difference(set(sales_train["item_id"])))

# There are no new shops in test set, but there are 363 new items! We can try use some constant prediction for them (like mean or median) or use global statistics for the particular shop.

# # Big picture

item_cnt_day_agg = sales_train.groupby("date")["item_cnt_day"].sum()

item_cnt_day_agg_smoothed = item_cnt_day_agg.rolling(window=7).mean()

fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(item_cnt_day_agg)
ax.plot(item_cnt_day_agg_smoothed, c="red")

item_cnt_month_agg = sales_train_by_month.groupby("date")["item_cnt_month"].sum()

fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(item_cnt_month_agg)

item_cnt_month_agg = item_cnt_month_agg.reset_index()
item_cnt_month_agg["year"] = item_cnt_month_agg["date"].dt.year
item_cnt_month_agg["month"] = item_cnt_month_agg["date"].dt.month

fig, ax = plt.subplots(figsize=(20, 8))
for year in item_cnt_month_agg["year"].unique():
    ax.plot(item_cnt_month_agg.loc[item_cnt_month_agg["year"] == year, "month"], item_cnt_month_agg.loc[item_cnt_month_agg["year"] == year, "item_cnt_month"], label=year)
ax.legend()

# The negative trend is apparent, with some seasonal peaks in March and December.

sales_train_by_month.describe(include="all")

# +
corr = sales_train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, center=0)
# -

plt.plot(sales_train['item_cnt_day'], '.')

test_sample = test.sample(frac=0.01)
sales_train_by_month_sample = sales_train_by_month.sample(frac=0.01)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(sales_train_by_month_sample["shop_id"], sales_train_by_month_sample["item_id"], 
           c=sales_train_by_month_sample["item_cnt_month"])
ax.scatter(test_sample["shop_id"], test_sample["item_id"], c="gray")
ax.set_xlabel("shop_id")
ax.set_ylabel("item_id")

item_cnt_by_category_id = sales_train_by_month.groupby("item_category_id")["item_cnt_month"].sum().reset_index().sort_values("item_cnt_month", ascending=False)
fig, ax = plt.subplots(figsize = (25, 5))
sns.barplot(x='item_category_id', y="item_cnt_month", data=item_cnt_by_category_id, order=item_cnt_by_category_id["item_category_id"], color="#1f77b4")

item_cnt_by_shop_id = sales_train_by_month.groupby("shop_id")["item_cnt_month"].sum().reset_index().sort_values("item_cnt_month", ascending=False)
fig, ax = plt.subplots(figsize = (25, 5))
sns.barplot(x="shop_id", y="item_cnt_month", data=item_cnt_by_shop_id, order=item_cnt_by_shop_id["shop_id"], color="#1f77b4")

# # Explore item counts

(sales_train["item_cnt_day"] == 0).value_counts(normalize=True)

# Training data don't consist explicit 0 - if there were no sales on the day, the row is absent.

sns.boxplot(sales_train["item_cnt_day"])

sales_train \
    .sort_values("item_cnt_day", ascending=False) \
    .head(20)

sales_train \
    .sort_values("item_cnt_day", ascending=False).head(20) \
    [["item_name", "item_name_en", "item_category_name", "item_category_name_en"]] \
    .drop_duplicates()

# - The top outlier seems to be some paid delivery service. The second top is the bag. 
# - The 3rd, 4th, 7th, 8th, 10th rows are the tickets for the same event (in two consecutive years) - IgroMir - which is some large-scale annual exhibition of computer and video games in Russia which in years 2013-2015 was held in October (tickets were probably sold much earlier). 
# - The other daily outliers were computer games (GTA, Middle-earth: Shadow of Mordor, The Witcher 3: Wild Hunt), iTunes payment card and Delivery Service (in Moscow only). This indicates that the number of items sold can vary according to some periodic (like IgroMir) or non-periodic events like the release of a popular computer game.

np.log(sales_train["item_cnt_day"]).hist()

np.log(sales_train["item_cnt_day"]).hist(bins = 30)

sales_train_by_month["item_cnt_cumsum"] = sales_train_by_month \
    .sort_values("date") \
    .groupby("item_id") \
    ["item_cnt_month"] \
    .cumsum()

df_item_cnt_cumsum = sales_train_by_month \
    .groupby(["date", "item_id"]) \
    ["item_cnt_month"] \
    .sum() \
    .groupby("item_id") \
    .cumsum() \
    .reset_index()

fig, ax = plt.subplots(figsize=(20, 12))
for i in df_item_cnt_cumsum["item_id"].unique():
    ax.plot(df_item_cnt_cumsum.loc[df_item_cnt_cumsum["item_id"] == i, "date"], 
            df_item_cnt_cumsum.loc[df_item_cnt_cumsum["item_id"] == i, "item_cnt_month"],
            label=i)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative sum of items sold")

# There's two things that strikes immedietaly from the plot:
#
# - There's one item that is sold predominantly (however, its predicted value be clipped to 20 anyway)
# - There are plenty of items that are no more in sold (apparent steady line at the end)

df_item_cnt_cumsum.loc[df_item_cnt_cumsum["item_cnt_month"] == df_item_cnt_cumsum["item_cnt_month"].max(),
                       "item_id"]

items.loc[items["item_id"] == 20949, "item_name"].values

# According to google translator this item is just a branded plastic bag, which makes total sense as it's probably sold almost with every order.

df_shop_cnt_cumsum = sales_train_by_month.groupby(["date", "shop_id"])["item_cnt_month"].sum().groupby("shop_id").cumsum().reset_index()

fig, ax = plt.subplots(figsize=(20, 12))
for i in df_shop_cnt_cumsum["shop_id"].unique():
    ax.plot(df_shop_cnt_cumsum.loc[df_shop_cnt_cumsum["shop_id"] == i, "date"], 
            df_shop_cnt_cumsum.loc[df_shop_cnt_cumsum["shop_id"] == i, "item_cnt_month"],
            label=i)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative sum of items sold by shop")

# There are some shops which are probably already closed. On other hand, some were opened after 2013.

# Based on two previous plots, we can probably predict 0 for all the items which are not sold anymore and shops that are closed (if they appear in the test set). However, these missing values could be for other reason as well. It'll be probably a good idea to see if these are the only missing values in the data or we have some missing values for another reason as well.

sales_train = sales_train.merge(test, how="left", on=["shop_id", "item_id"])

needed_combinations = pd.DataFrame(list(itertools.product(sales_train_by_month["date"].unique(), test["ID"].unique())),
                                   columns=["date", "ID"])

needed_combinations = needed_combinations.merge(sales_train_by_month, how="left", on=["date", "ID"])

needed_combinations.isnull().sum(axis=0)

needed_combinations.shape

needed_combinations.dtypes

needed_combinations["date"] = needed_combinations["date"].dt.date

item_cnt_month_by_item = needed_combinations.pivot_table(values="item_cnt_month", index="item_id", columns="date")

fig, ax = plt.subplots(figsize=(20, 50))
ax = sns.heatmap(item_cnt_month_by_item.notnull(), cmap=sns.color_palette(n_colors=2))

# Plenty of missing values with irregular patterns, but it seems that some items were released after the data collection started and some of them are not sold anymore.

item_cnt_month_by_item = item_cnt_month_by_item.loc[item_cnt_month_by_item.apply(lambda row: row.first_valid_index(), axis=1).sort_values().index, :]

fig, ax = plt.subplots(figsize=(20, 50))
ax = sns.heatmap(item_cnt_month_by_item.notnull(), cmap=sns.color_palette(n_colors=2))

item_cnt_month_by_item = item_cnt_month_by_item.loc[item_cnt_month_by_item.apply(lambda row: row.last_valid_index(), axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(20, 50))
ax = sns.heatmap(item_cnt_month_by_item.notnull(), cmap=sns.color_palette(n_colors=2))

# There is some parts of items that is probably not sold anymore. However, some items were sold again even after few months break. This means that predicting 0 for items that seems to be not in sale for a couple of months isn't probably the best solution. However, the features like first valid index and last valid index can be useful.

sales_train_by_month.loc[sales_train_by_month["item_cnt_month"] < 0, "item_cnt_month"].value_counts()

sales_train.loc[sales_train["item_cnt_day"] < 0, "item_cnt_day"].value_counts()

# These are probably returned items. We can either remove those negative values (set to 0), but I believe it would create a bias (as if more items are sold than returned the value is positive and those returned items are still present in total value). Probably better to leave them as they are.

np.log1p(sales_train_by_month.loc[sales_train_by_month["item_cnt_month"] >= 0, "item_cnt_month"]).hist(bins=20)

needed_combinations.groupby("date")["item_cnt_month"].agg(lambda x: x.isnull().sum() / len(x))

# # Explore the price

sales_train["item_price"].value_counts()

(100 * sales_train["item_price"] % 1).value_counts()

((100 * sales_train["item_price"] % 1) > 0).value_counts(normalize=True)

# Over 99% of products has a price which is an integer value, but there are some items with irregular decimals. I believe it may be an indication of some discount applied and therefore, and indication of occasional promotion.

irregular_decimals = (100 * sales_train["item_price"] % 1) > 0

irregular_decimals_dates = sales_train.loc[irregular_decimals, "date"].drop_duplicates()

(irregular_decimals_dates - irregular_decimals_dates.shift(1, fill_value="2012-12-31")).value_counts()

np.log(sales_train["item_price"]).hist()

sales_train.loc[sales_train["item_price"] < 0]

# Seems like invalid value. We can deal with that later with some imputation method.

sales_train.loc[sales_train["item_id"] == 0]

sales_train["item_price"].describe(percentiles=np.arange(0.1, 1, 0.1)).apply(lambda x: format(x, 'g'))

sales_train.sort_values("item_price", ascending=False).head(20)

# The most expensive item was Radmin3 license sold probably in a package for 522 people. The second top item with respect to price was a delivery, which is a little bit weird, but feasible in case of an overseas delivery of the large order. Other most expensive items were either game consoles or highly-technical software.

item_price_stats = sales_train.groupby("item_id") \
    .agg({"item_price": [np.nanstd, np.mean, np.median, "count"], "shop_id": "nunique"})
item_price_stats.rename({"nunique": "n_shops"}, axis=1, inplace=True)
item_price_stats = item_price_stats.T.reset_index(level=0, drop=True).T
item_price_stats["dispersion"] = item_price_stats["nanstd"] ** 2 / item_price_stats["mean"]
item_price_stats.sort_values("dispersion", ascending=False).head(30)

sales_train.groupby("item_id")["shop_id"].nunique()

# There are a lot of items with really large dispersion from the mean value. Let's inspect few of them.

items[items["item_id"] == 11365]

fig, ax = plt.subplots(figsize=(20, 5))
for id, shop_df in sales_train.sort_values("date").loc[sales_train["item_id"] == 11365].groupby('shop_id'):
    ax.plot(shop_df["date"],
            shop_df["item_price"], 
            "o-", label=id)
ax.legend()

# This is certainly weird distribution of item price. There's few outliers so median value should be probably used if we want to aggregate values of `item_price`.

fig, ax = plt.subplots(figsize=(20, 5))
for id, shop_df in sales_train.sort_values("date").loc[sales_train["item_id"] == 13477].groupby('shop_id'):
    ax.plot(shop_df["date"],
            shop_df["item_price"], 
            "o-", label=id)
ax.legend()

sales_train.sort_values("date").loc[sales_train["item_id"] == 13477]

# It's unlikely that the price was nearly doubled in less than two weaks, but maybe there were some promotion applied. 

fig, ax = plt.subplots(figsize=(20, 5))
for id, shop_df in sales_train.sort_values("date").loc[sales_train["item_id"] == 16854].groupby('shop_id'):
    ax.plot(shop_df["date"],
            shop_df["item_price"], 
            "o-", label=id)
ax.legend()

# Price of the item 16854 varies not only accross shops but also within the same shops.

fig, ax = plt.subplots(figsize=(20, 5))
for id, shop_df in sales_train.sort_values("date").loc[sales_train["item_id"] == 11370].groupby('shop_id'):
    ax.plot(shop_df["date"],
            shop_df["item_price"], 
            "o-", label=id)
ax.legend()

# As with first item 11365, we observe outliers in price.

sales_train.loc[sales_train["item_id"] == 11370, "shop_id"].drop_duplicates()
