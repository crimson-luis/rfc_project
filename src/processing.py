from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from yaml import safe_load
import pandas as pd
import numpy as np


with open("./config.yaml", "r") as file:
    config = safe_load(file)


def generate_folds(df):
    y = df[config["DATASET"]["CLASS"]]
    n_samples = df.shape[0]
    skf = StratifiedKFold(n_splits=config["DATASET"]["N_SPLITS"])
    i = 1
    for train, test in skf.split(np.zeros(n_samples), y):
        df.iloc[train.astype(int)].to_csv(
            f"./data/processed/{config['DATASET']['NAME']}.train{i}.csv", index=False)
        df.iloc[test.astype(int)].to_csv(
            f"./data/processed/{config['DATASET']['NAME']}.test{i}.csv", index=False)
        i += 1


def encode(x, y, save: bool = False):
    enc = OneHotEncoder().fit(x)
    x = pd.DataFrame(
        data=enc.transform(x).toarray(),
        columns=enc.get_feature_names(np.array(list(x.columns)))
    )
    df = pd.concat([y, x], axis=1)
    df[config["DATASET"]["CLASS"]] = pd.factorize(y, sort=True)[0]
    if save:
        df.to_csv(
            f"./data/processed/{config['DATASET']['NAME']}.full.csv",
            index=False
        )
    return df

#
# def factorize_data(dataset: pd.DataFrame, show: bool = True):
#     columns = dataset.columns
#     print(f"{columns}\n")
#     factorized_dataframe = pd.DataFrame()
#     for column in columns:
#         factorized_column = pd.factorize(dataset[column], sort=True)[0]
#         factorized_dataframe[f"{column}_f"] = factorized_column
#         check_unique = pd.DataFrame(
#             [factorized_column, dataset[column]],
#             index=[f"{column}_f", f"{column}"]
#         )
#         if show:
#             print(check_unique.T.drop_duplicates(
#                 subset=[f"{column}_f"]
#             ).sort_values(by=[column]))
#     return factorized_dataframe
