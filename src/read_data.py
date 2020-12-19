import pandas as pd
import numpy as np


def read_data():

    sales_train = pd.read_csv("../input/sales_train.csv",
                              dtype={"date_block_num": np.int8,
                                      "shop_id": np.int8,
                                      "item_id": np.int16,
                                      "item_price": np.float32,
                                      "item_cnt_day": np.int32},
                              parse_dates=["date"],
                              dayfirst=True)

    test = pd.read_csv("../input/test.csv",
                       dtype={"ID": np.int32,
                              "shop_id": np.int8,
                              "item_id": np.int16})

    try:
        items = pd.read_csv("../auxiliaries/items_translated.csv",
                            dtype={"item_id": np.int16,
                                   "item_category_id": np.int8})
    except FileNotFoundError:
        items = pd.read_csv("../input/items.csv",
                            dtype={"item_id": np.int16,
                                   "item_category_id": np.int8})

    try:
        item_categories = pd.read_csv("../auxiliaries/item_categories_translated.csv",
                                      dtype={"item_category_id": np.int8})
    except FileNotFoundError:
        item_categories = pd.read_csv("../input/item_categories.csv",
                                      dtype={"item_category_id": np.int8})

    try:
        shops = pd.read_csv("../auxiliaries/shops_translated.csv",
                            dtype={"shop_id": np.int8})
    except FileNotFoundError:
        shops = pd.read_csv("../input/shops.csv",
                            dtype={"shop_id": np.int8})

    calendar = pd.read_csv("../auxiliaries/bank_holidays_calendar.csv",
                           dtype={"weekday": np.int8},
                           parse_dates=["date"])

    return sales_train, test, items, item_categories, shops, calendar


if __name__ == "__main__":

    sales_train, test, items, item_categories, shops, calendar = read_data()
