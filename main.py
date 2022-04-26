from src.random_forest_classifier import GaussianModel
from sklearn.ensemble import RandomForestClassifier
from src.processing import factorize_data
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# TODO: - usar métodos para ranquear as features;
#       - diminuir o número de features para no mínimo umas 20, após a codificação;
#       - separar o código em funções;
#       - quais caracteristicas teria um dataset para aplicar o BAT?
#       - olhar no livro do README;
DATASET = "STULONG"
# Sugestão da quantidade de nós: de 1 à p ** (1 / 2).
# https://towardsdatascience.com/frequent-pattern-mining-association-and-correlations-8fa9f80c22ef
PATH = "C:/Users/luisg/OneDrive/Documentos/Python_Scripts/rfc_project"
# https://lispminer.vse.cz/demonstration/stulong/skupina.html
cardio_df = pd.read_csv(f"{PATH}/data/raw/{DATASET}.full.csv")

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
label = cardio_df.columns[0]
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
x = pd.DataFrame(
    data=enc.transform(x).toarray(),
    columns=enc.get_feature_names(features)
)
cardio_df = pd.concat([y, x], axis=1)
cardio_df.replace({"A": 0.0, "B": 1.0, "C": 2.0}, inplace=True)
cardio_df.to_csv(f"{PATH}/data/processed/{DATASET}.full.csv", index=False)
feature_list = list(x.columns)
cardio_df = pd.read_csv(f"{PATH}/data/processed/{DATASET}.full.csv")

# KFold, Repeated KFold,  Stratified KFold, Group k-fold
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x, y)
i = 1
for train, test in skf.split(x, y):
    cardio_df.iloc[train.astype(int)].to_csv(
        f"./data/processed/{DATASET}.train{i}.csv", index=False)
    cardio_df.iloc[test.astype(int)].to_csv(
        f"./data/processed/{DATASET}.test{i}.csv", index=False)
    i += 1


# salvando as árvores
def model_to_txt(index, model, show: bool = True, save: bool = False):
    # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    forest = model.estimators_
    model_info = list()
    header = (f"DATASET_NAME: {DATASET}.train{index}.csv"
              f"\nENSEMBLE: RF"
              f"\nNB_TREES: {len(forest)}"
              f"\nNB_FEATURES: {forest[0].tree_.n_features}"
              f"\nNB_CLASSES: {forest[0].tree_.max_n_classes}"
              f"\nMAX_TREE_DEPTH: {forest[0].tree_.max_depth}"
              "\nFormat: node / node type (LN - leave node, IN - internal node) "
              "left child / right child / feature / threshold / node_depth / "
              "majority class (starts with index 0)")
    model_info.append(header)
    for tree_idx, est in enumerate(forest):
        tree = est.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        tree_title = f"\n\n[TREE {tree_idx}]\nNB_NODES: {n_nodes}"
        model_info.append(tree_title)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        for i in range(n_nodes):
            class_idx = np.argmax(value[i][0])
            if is_leaves[i]:
                tree_txt = f"\n{i} LN -1 -1 -1 -1 {node_depth[i]} {class_idx}"
            else:
                tree_txt = (f"\n{i} IN {children_left[i]} {children_right[i]} "
                            f"{feature[i]} {threshold[i]} {node_depth[i]} -1")
            model_info.append(tree_txt)
    if show:
        print(*model_info)
    if save:
        with open(f"{PATH}/data/processed/forests/{DATASET}.RF{index}.txt", "w") as f:
            for item in model_info:
                f.write(item)


j = int
for j in range(1, 10):
    print(f"{j}. reading test and train.")
    test = pd.read_csv(f"{PATH}/data/processed/{DATASET}.test{j}.csv")
    train = pd.read_csv(f"{PATH}/data/processed/{DATASET}.train{j}.csv")
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
    # rf = RandomForestClassifier(
    #     random_state=0,
    #     max_depth=2,
    # )
    rf.fit(train_features, train_labels)
    rf.score(train_features, train_labels)

    # ou seja, chutou tudo 1.
    predictions = pd.Series(rf.predict(test_features))
    print(f"{predictions.value_counts()}")

    # Previsões totais corretas da classe
    # Se a previsão for correta a diferença test_label - predictions será zero.
    print(confusion_matrix(test_labels, predictions))
    model_to_txt(index=j, model=rf, save=True)



