import pandas as pd
from googletrans import Translator
from time import sleep

def ru_to_en(text):
    translator = Translator()
    while True:
        try:
            return translator.translate(text, src="ru", dest="en").text
        except:
            print("I need a break. Will try again in 30 minutes.")
            sleep(3600)
            translator = Translator()


def ru_to_pl(text):
    translator = Translator()
    while True:
        try:
            return translator.translate(text, src="ru", dest="pl").text
        except:
            print("I need a break. Will try again in 30 minutes.")
            sleep(3600)
            translator = Translator()


if __name__ == '__main__':
    item_categories = pd.read_csv("./input/item_categories.csv")
    items = pd.read_csv("./input/items.csv")

    item_categories["item_category_name_en"] = item_categories["item_category_name"] \
        .apply(lambda x: ru_to_en(x))
    item_categories["item_category_name_pl"] = item_categories["item_category_name"] \
        .apply(lambda x: ru_to_pl(x))

    items["item_name_en"] = items["item_name"] \
        .apply(lambda x: ru_to_en(x))
    items["item_name_pl"] = item_categories["item_name"] \
        .apply(lambda x: ru_to_pl(x))

    item_categories.to_csv("./auxiliaries/item_categories_translated.csv")
    items.to_csv("./auxiliaries/items_translated.csv")
