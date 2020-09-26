import pandas as pd

sales_train = pd.read_csv("../input/sales_train.csv")
test = pd.read_csv("../input/test.csv")
shops = pd.read_csv("../input/shops.csv")

try:
    items = pd.read_csv("../auxiliaries/items_translated.csv")
except FileNotFoundError:
    items = pd.read_csv("../input/items.csv")

try:
    item_categories = pd.read_csv("../auxiliaries/item_categories_translated.csv")
except FileNotFoundError:
    item_categories = pd.read_csv("../input/item_categories.csv")
