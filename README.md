# Predict Future Sales - Kaggle competition

This repository contains my solution to [Kaggle competition](
https://www.kaggle.com/c/competitive-data-science-predict-future-sales/), in which the participants were provided with time-series dataset from a Russian software company and asked to predict total sales for every product and store in the next month.

#### EDA and Data Preprocessing
Exploratory Data Analysis can be found under `notebooks/eda.ipynb`. The significant findings include:

- training set includes only non-zero sales, while the test set consists of all shop/item combinations; therefore a proper upsampling is necessary before training;
- as usual with time-series data, there are some immediately visible seasonality patterns;
- names of items, item categories or shop reveal underlying, less granular categorisation or other information that can be further utilised;
- some shops exists under two separate IDs;
- there are some outliers in the dataset;
- some shops are probably already closed, and some items are probably not in sale anymore.

All of this drove the data preprocessing and feature engineering stages. See `src/preprocess_data.py` for details of data preprocessing. 

### Feature Engineering
The feature engineering process involved:

- Manually extracted features from the description of items/item categories/shops
- Entity embeddings for items/item categories/shops (see `src/train_embeddings.py` for PyTorch model specification)
- Calendar features like no. of weekends, bank holidays etc. within each month
- Lagged aggregates of sales and prices as well as possible discounts, grouped by any combination of item/item category/shop apart from item \cross item category as the latest is naturally significantly correlated
- Rolling window aggregates of sales grouped as above
- Missingness patterns that can exhibit, for example, information that an item is not in sale anymore

Only the subset of created features was used in the final models.

#### Cross-validation

The cross-validation procedure was designed to deal with time-dependent panel data. `GroupedTimeSeriesSplit` (see  `src/utils.py` for details) was therefore applied in two ways:

- full cross-validation, with months 1-33 subsequently serving as a validation set and all months prior to the current validation set used for training;
- simple cross-validation, with 33rd month used as a validation set and months 0-32 utilised during training.
The latter was favoured due to computational speed and comparable correlation with the test-set error.

#### Modelling

Due to the tabular nature of data, I decided to utilise boosting trees algorithms that usually perform extremely well in these settings: XGBoost and CatBoost. Their hyperparameters were tuned with Bayesian optimisation using `scikit-optimize`.

#### Result

Those scripts placed me in top 8% (778/10,159 teams) at the time of writing, with public set RMSE of 0.89232. To obtain the same score (or higher!), you'd need to choose a subset of features, tweak the hyperparameters, use some form of ensembling and post-process the predictions.