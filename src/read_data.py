import pandas as pd
import numpy as np


def read_data():

    sales_train = pd.read_csv("../input/sales_train.csv")
    sales_train["date"] = pd.to_datetime(sales_train["date"], format="%d.%m.%Y")
    sales_train = sales_train.astype({"date_block_num": np.int8, "shop_id": np.int8, "item_id": np.int16,
                                      "item_price": np.float32, "item_cnt_day": np.int32})

    test = pd.read_csv("../input/test.csv")

    try:
        items = pd.read_csv("../auxiliaries/items_translated.csv")
    except FileNotFoundError:
        items = pd.read_csv("../input/items.csv")

    try:
        item_categories = pd.read_csv("../auxiliaries/item_categories_translated.csv")
    except FileNotFoundError:
        item_categories = pd.read_csv("../input/item_categories.csv")

    try:
        shops = pd.read_csv("../auxiliaries/shops_translated.csv")
    except FileNotFoundError:
        shops = pd.read_csv("../input/shops.csv")

    calendar = pd.read_csv("../auxiliaries/bank_holidays_calendar.csv")

    return sales_train, test, items, item_categories, shops, calendar


if __name__ == "__main__":

    sales_train, test, items, item_categories, shops, calendar = read_data()
