from src.random_forest_classifier import RFClassifier
from src.processing import encode, generate_folds
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import yaml

# Dataset Table.
# B - Binário; D - Categorias; F - Número.
# Feature list.
# Data  set             n       p   K   CD          Src
# BC:   Breast-Cancer   683     9   2   65-35       UCI
# CP:   COMPAS          6907    12  2   54-46       HuEtAl
# FI:   FICO            10459   17  2   52-48       HuEtAl
# HT:   HTRU2           17898   8   2   91-9        UCI
# PD:   Pima-Diabetes   768     8   2   65-35       SmithEtAl
# SE:   Seeds           210     7   3   33-33-33    UCI
# ST:   STULONG         788     182 3   72-21-7     UCI
# split into x and y. Train and test. 1/10 within 10 folds.

# TODO: - usar métodos para ranquear as features;
#       - diminuir o número de features para no mínimo umas 20, após a codificação;
#       - separar o código em funções;
#       - quais caracteristicas teria um dataset para aplicar o BAT?
#       - olhar no livro do README;
#       - testar juntando os grupos de doença. A e B ou B e C.
#       - fazer um exp com CART.
#       - comparar resultado preditivo e interpretabilidade.

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Sugestão da quantidade de nós: de 1 à p ** (1 / 2).
# https://towardsdatascience.com/frequent-pattern-mining-association-and-correlations-8fa9f80c22ef
# https://lispminer.vse.cz/demonstration/stulong/skupina.html
DATASET = config['DATASET']['NAME']
cardio_df = pd.read_csv(f"./data/raw/{DATASET}.full.csv")
label = config['DATASET']['CLASS']

# A: Normal Group
# B: Risk Group
# C: Pathologic Group
cardio_df = cardio_df[[
    "BasicGroup",
    "GroupCode",
    "AgeRange",
    "Smoking",
    "WeightRange",
    "BMIRange",
    "BloodPressure",
    "SkinfoldSubsc",
    "SkinfoldTric",
    "Cholesterol",
    "Triglycerides",
]]
# Samples containing missing values have been eliminated (420 occurrences)
cardio_df = cardio_df.replace("NS", np.NAN).dropna().reset_index(drop=True)
describe = cardio_df.describe()
cardio_df.SkinfoldSubsc = pd.to_numeric(cardio_df.SkinfoldSubsc, errors="coerce")
cardio_df.SkinfoldTric = pd.to_numeric(cardio_df.SkinfoldTric, errors="coerce")
cardio_df["Skinfold"] = cardio_df.SkinfoldSubsc + cardio_df.SkinfoldTric
cardio_df.drop(columns=["SkinfoldSubsc", "SkinfoldTric"], inplace=True)
cardio_df["Skinfold"] = np.where(
    cardio_df.Skinfold < 20,
    "8 - 20",  # 8 - 20
    np.where(
        cardio_df.Skinfold < 30,
        "21 - 30",  # 21 - 30
        np.where(
            cardio_df.Skinfold < 40,
            "31 - 40",  # 31 - 40
            "> 40"  # > 40
        )
    )
)

x = cardio_df.drop(label, axis=1)
y = cardio_df[label]
feature_list = list(x.columns)
features = np.array(feature_list)
labels = np.array(cardio_df[label])

# Codificando as colunas em features binárias. 182 features, K = 3.
# Classes are always indexed from 0 to K-1.
# cardio_df = encode(x, y)
feature_list = list(x.columns)
cardio_df = pd.read_csv(f"./data/processed/{DATASET}.full.csv")
label = cardio_df.columns[0]

# Stratified KFold
# generate_folds(cardio_df)

j = int
for j in range(1, 11):
    print(f"{j}. Lendo test e train.\n")
    test = pd.read_csv(f"./data/processed/{DATASET}.test{j}.csv")
    train = pd.read_csv(f"./data/processed/{DATASET}.train{j}.csv")
    train_features = train.drop([label], axis=1)
    train_labels = train[label]

    rf = RFClassifier()
    rf.fit(train_features, train_labels)
    rf.score(train_features, train_labels)

    predictions = pd.Series(rf.predict(test.drop([label], axis=1)))
    print(predictions.value_counts())

    rf.model_to_txt(index=j, show=True)
    print(confusion_matrix(test[label], predictions))



