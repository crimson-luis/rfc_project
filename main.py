from src.random_forest_classifier import GaussianModel
from sklearn.ensemble import RandomForestClassifier
from src.processing import factorize_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# TODO: - avaliar quais variáveis são relevantes;
#       - usar métodos para ranquear as features;
#       - diminuir o número de features para no mínimo umas 20, após a codificação;
#       - aplicar o algoritmo para as 10 dobras do dataset;
#       - separar o código em funções;
#       - quais caracteristicas teria um dataset para aplicar o BAT?
#       - olhar no livro do README;

# Sugestão da quantidade de nós: de 1 à p ** (1 / 2).
# https://towardsdatascience.com/frequent-pattern-mining-association-and-correlations-8fa9f80c22ef
PATH = "C:/Users/luisg/OneDrive/Documentos/Python_Scripts/rfc_project"
cardio_df = pd.read_csv(f"{PATH}/data/raw/OneHot/STULONG.full.csv")

# A: Normal Group
# B: Risk Group
# C: Pathologic Group
label = cardio_df.columns[0]
# Samples containing missing values have been eliminated (420 occurrences)
cardio_df = cardio_df.replace("NS", np.NAN).dropna().reset_index(drop=True)
describe = cardio_df.describe()

# mostrar um exemplo de uma observação no documento.

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

x = cardio_df.drop(label, axis=1)
y = cardio_df[label]
feature_list = list(x.columns)
features = np.array(feature_list)
labels = np.array(cardio_df[label])

# Codificando as colunas em features binárias. 182 features, K = 3.
# Classes are always indexed from 0 to K-1.
enc = OneHotEncoder()
enc.fit(x)
print(enc.categories_)
x_ohe = pd.DataFrame(
    data=enc.transform(x).toarray(),
    columns=enc.get_feature_names(features)
)
cardio_df_ohe = pd.concat([y, x_ohe], axis=1)
cardio_df_ohe.replace({"A": 0.0, "B": 1.0, "C": 2.0}, inplace=True)
cardio_df_ohe.to_csv(f"{PATH}/data/raw/OneHot/STULONG.full.csv", index=False)
feature_list = list(x_ohe.columns)

# KFold, Repeated KFold,  Stratified KFold, Group k-fold
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_ohe, y)
i = 1
for train, test in skf.split(x_ohe, y):
    cardio_df_ohe.iloc[train.astype(int)].to_csv(
        f"./data/raw/OneHot/STULONG.train{i}.csv", index=False)
    cardio_df_ohe.iloc[test.astype(int)].to_csv(
        f"./data/raw/OneHot/STULONG.test{i}.csv", index=False)
    i += 1

# Rodando um classificador Gaussiano - RF - para cada fold.
test = pd.read_csv(f"{PATH}/data/raw/OneHot/STULONG.test1.csv")
train = pd.read_csv(f"{PATH}/data/raw/OneHot/STULONG.train1.csv")
train_features = train.drop([label], axis=1)
train_labels = train[label]
test_features = test.drop([label], axis=1)
test_labels = test[label]

rf = RandomForestClassifier(
    n_estimators=10,
    random_state=42,
    max_depth=3,
    max_leaf_nodes=8,

)
rf.fit(train_features, train_labels)
rf.score(train_features, train_labels)

# ou seja, chutou tudo 1.
predictions = pd.Series(rf.predict(test_features))

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(
    tree,
    out_file=f"{PATH}/tree.dot",
    feature_names=feature_list,
    rounded=True,
    precision=1
)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file(f"{PATH}/tree.dot")
# Write graph to a png file
graph.write_png(f"{PATH}/tree.png")

