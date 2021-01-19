<<<<<<< HEAD
import argparse
=======
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
<<<<<<< HEAD
from collections import namedtuple, OrderedDict
from itertools import product
from preprocess_data import read_preprocessed_data
from utils import clip_target


class EmbeddingNet(nn.Module):
    def __init__(self,
                 items, item_categories, shops,
                 embed_size={"items": 24,
                             "item_categories": 8,
                             "shops": 8},
                 n_epochs=3,
                 lr=1e-3,
                 batch_size=1024,
                 dropout_rate=0.2,
=======
from preprocess_data import read_preprocessed_data


class EmbeddingNet(nn.Module):
    def __init__(self, 
                 items, item_categories, shops, 
                 embed_size={"items": 64, "item_categories": 8, "shops": 8}, 
                 n_epochs=3,
                 lr=1e-2,
                 batch_size=128,
                 dropout_rate=0.3,
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a
                 device="cuda"):

        super().__init__()

        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.device = torch.device(device)

        embed_input_size = embed_size["items"] + embed_size["item_categories"] + embed_size["shops"]

<<<<<<< HEAD
        self.item_embeddings = nn.Embedding(num_embeddings=items.shape[0],
                                            embedding_dim=embed_size["items"])
        self.item_category_embeddings = nn.Embedding(num_embeddings=item_categories.shape[0],
                                                     embedding_dim=embed_size["item_categories"])
        self.shop_embeddings = nn.Embedding(num_embeddings=shops.shape[0],
                                            embedding_dim=embed_size["shops"])
        self.dropout0 = nn.Dropout(p=self.dropout_rate)
        self.batch_norm0 = nn.BatchNorm1d(num_features=embed_input_size)

        self.fc1 = nn.Linear(in_features=embed_input_size,
                             out_features=np.ceil((embed_input_size) / 2).astype("int"))
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(num_features=np.ceil((embed_input_size) / 2).astype("int"))

        self.fc2 = nn.Linear(in_features=np.ceil((embed_input_size) / 2).astype("int"),
                             out_features=np.ceil((embed_input_size) / 4).astype("int"))
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.batch_norm2 = nn.BatchNorm1d(num_features=np.ceil((embed_input_size) / 4).astype("int"))

        self.out = nn.Linear(in_features=np.ceil((embed_input_size) / 4).astype("int"),
                             out_features=1)


    def forward(self, input):

=======
        self.item_embeddings = nn.Embedding(items.shape[0], embed_size["items"])
        self.item_category_embeddings = nn.Embedding(item_categories.shape[0], embed_size["item_categories"])
        self.shop_embeddings = nn.Embedding(shops.shape[0], embed_size["shops"])
        self.dropout0 = nn.Dropout(p=self.dropout_rate)
        self.batch_norm0 = nn.BatchNorm1d(num_features=embed_input_size)

        self.fc1 = nn.Linear(embed_input_size,
                             np.ceil((embed_input_size) / 2).astype("int"))
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(num_features=np.ceil((embed_input_size) / 2).astype("int"))

        self.fc2 = nn.Linear(np.ceil((embed_input_size) / 2).astype("int"),
                             np.ceil((embed_input_size) / 4).astype("int"))
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.batch_norm2 = nn.BatchNorm1d(num_features=np.ceil((embed_input_size) / 4).astype("int"))

        self.out = nn.Linear(np.ceil((embed_input_size) / 4).astype("int"), 
                             1)


    def forward(self, input):
        
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a
        t = input
        t1 = self.item_embeddings(t[:, 0].long())
        t2 = self.item_category_embeddings(t[:, 1].long())
        t3 = self.shop_embeddings(t[:, 2].long())
        t = torch.cat((t1, t2, t3), dim=1)
        t = self.dropout0(t)
        t = self.batch_norm0(t)
<<<<<<< HEAD

        t = F.relu(self.fc1(t))
        t = self.dropout1(t)
        t = self.batch_norm1(t)

        t = F.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.batch_norm2(t)

        t = self.out(t)

=======
        
        t = F.relu(self.fc1(t))
        t = self.dropout1(t)
        t = self.batch_norm1(t)
        
        t = F.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.batch_norm2(t)
        
        t = self.out(t)
        
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a
        return t.squeeze()


    def preprocess_data(self, X, y=None):
        lbl_enc = LabelEncoder()
        X.loc[:, "shop_id"] = lbl_enc.fit_transform(X.loc[:, "shop_id"])
        X = torch.tensor(X[["item_id", "item_category_id", "shop_id"]].values).to(device=self.device)
        if y is not None:
            y = torch.tensor(y.values).to(device=self.device)
            return torch.utils.data.TensorDataset(X, y)
        else:
            return torch.utils.data.TensorDataset(X)


    def fit(self, X_train, y_train, eval_set=None):

        train_set = self.preprocess_data(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)

        for epoch in range(self.n_epochs):
<<<<<<< HEAD

            train_loss = 0.0
            for inputs, labels in tqdm(train_loader):
=======
            
            total_loss = 0.0
            running_loss = 0.0
            for batch_id, (inputs, labels) in enumerate(tqdm(train_loader)):
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a

                preds = self.forward(inputs)
                loss = F.mse_loss(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

<<<<<<< HEAD
                train_loss += loss.item()

            train_loss = np.sqrt(train_loss / len(train_loader))
            print(f"epoch: {epoch} \t train loss: \t {train_loss}")
=======
                total_loss += loss.item()
                running_loss += loss.item()

                if (batch_id + 1) % (len(train_set) / self.batch_size // 10) == 0:
                    print(f"epoch {epoch}, batch {batch_id + 1}, loss: {np.sqrt(running_loss / (len(train_set) / self.batch_size // 10))}")
                    running_loss = 0.0                    

            total_loss = np.sqrt(total_loss / len(train_loader))
            print(f"epoch: {epoch} \t train loss: \t {total_loss}")
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a

            if eval_set is not None:
                X_val, y_val = eval_set
                preds = self.predict(X_val)
<<<<<<< HEAD
                eval_loss = torch.sqrt(F.mse_loss(preds, torch.tensor(y_val.values, device=self.device)))
                print(f"epoch: {epoch} \t validation loss: \t {eval_loss}")
=======
                print(f"epoch: {epoch} \t validation loss: \t \
                        {torch.sqrt(F.mse_loss(preds, torch.tensor(y_val.values, device=self.device)))}")
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a


    def predict(self, X):
        dataset = self.preprocess_data(X)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4096)

        self.eval()

        preds = []
        for batch in data_loader:
            preds.append(self.forward(batch[0]))
        preds = torch.cat(preds)

        self.train()

        return preds


<<<<<<< HEAD
class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train entity embeddings for item_id, item_category_id and shop_id.")
    parser.add_argument("-t", "--tune",
                        dest="tune", action="store_true",
                        help='Whether to tune hyperparameters.')
    args = parser.parse_args()
    print(args)

    sales_by_month, _, _, items, item_categories, shops, _ = read_preprocessed_data()
                                                                                                                                                                                                                                                                                                                                                                        
    X = sales_by_month.loc[sales_by_month["date_block_num"] < 34].drop("item_cnt_month", axis=1).copy()
    y = sales_by_month.loc[sales_by_month["date_block_num"] < 34, "item_cnt_month"].copy()
    y = clip_target(y)

    train_id = sales_by_month[sales_by_month["date_block_num"] < 33].index
    val_id = sales_by_month[sales_by_month["date_block_num"] == 33].index
    X_train = X.iloc[train_id, :]
    y_train = y.iloc[train_id]
    X_val = X.iloc[val_id, :]
    y_val = y.iloc[val_id]

    if args.tune:
        params = OrderedDict(
            lr=[1e-3, 1e-4, 1e-5],
            batch_size=[1024],
            dropout_rate=[0.1, 0.2, 0.3]
        )
        for run in RunBuilder.get_runs(params):
            print(run)
            embed = EmbeddingNet(items, item_categories, shops,
                                 embed_size={"items": 16,
                                             "item_categories": 8,
                                             "shops": 8},
                                 n_epochs=3,
                                 lr=run.lr,
                                 batch_size=run.batch_size,
                                 dropout_rate=run.dropout_rate,
                                 device="cuda")
            embed.fit(X_train, y_train,
                      eval_set=(X_val, y_val))
    else:
        embed = EmbeddingNet(items, item_categories, shops,
                             embed_size={"items": 16,
                                         "item_categories": 8,
                                         "shops": 8},
                             n_epochs=3,
                             lr=1e-4,
                             batch_size=1024,
                             dropout_rate=0.3,
                             device="cuda")
        embed.fit(X_train, y_train,
                  eval_set=(X_val, y_val))

        torch.save(embed.state_dict(),
                   f"../models/embedding_model_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
=======
if __name__ == "__main__":

    sales_by_month, _, _, items, item_categories, shops, _ = read_preprocessed_data()

    X = sales_by_month.loc[sales_by_month["date_block_num"] < 34].drop("item_cnt_month", axis=1).copy()
    y = sales_by_month.loc[sales_by_month["date_block_num"] < 34, "item_cnt_month"].copy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    embed = EmbeddingNet(items, item_categories, shops,
                         embed_size={"items": 16, "item_categories": 8, "shops": 8},
                         n_epochs=5,
                         lr=1e-2,
                         batch_size=1024,
                         dropout_rate=0.2,
                         device="cuda")
    embed.fit(X_train, y_train,
              eval_set=(X_val, y_val))

    torch.save(embed, 
               f"../models/embedding_model_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
>>>>>>> dc564696db0136ea4645b6e5268d864652a12f4a
