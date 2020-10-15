import pandas as pd


def read_data():

    sales_train = pd.read_csv("../input/sales_train.csv")
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
