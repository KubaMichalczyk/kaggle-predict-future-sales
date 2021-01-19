import pandas as pd
from pyarrow.parquet import read_schema

ITEM_FEATURES = True
SHOP_FEATURES = True
ITEM_CATEGORY_FEATURES = True
CALENDAR_FEATURES = True
LAGGED_FEATURES = True
ROLLING_FEATURES = True
MISSINGNESS_FEATURES = True
EMBEDDING_FEATURES = True

MEDIAN_FEATURES = True
MEAN_FEATURES = True
BY_SHOP_ID = True
BY_ITEM_ID = True
BY_ITEM_CATEGORY_ID = True
BY_SHOP_ID_ITEM_ID = True
BY_SHOP_ID_ITEM_CATEGORY_ID = True

DATA_FILE = "../input/all_features.parquet"
FEATURES = pd.Index(read_schema(DATA_FILE).names).drop(["__index_level_0__", "item_cnt_month"], errors="ignore")
SELECTED_FEATURES = None
EARLY_STOPPING_ROUNDS = 20
