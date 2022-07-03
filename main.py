# from sklearn.inspection import permutation_importance
# from src.random_forest_classifier import RFClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from pickle import dump, load
# from sklearn.model_selection import RandomizedSearchCV
from src.processing import (
    encode,
    generate_folds,
    transform,
    model_to_txt,
    normalized_confusion_matrix,
    plot_confusion_matrix,
)
from sklearn.metrics import confusion_matrix, precision_score
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
import persistence as pr
from sklearn import tree  # for decision tree models
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

# TODO:
#       - diminuir o número de features para no maximo umas 20, após a codificação;
#       - quais caracteristicas teria um dataset para aplicar o BAT?
#       - olhar no livro do README;
#       - testar juntando os grupos de doença. A e B ou B e C.
#       - comparar resultado preditivo e interpretabilidade.
#       - atualizar poetry.

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Sugestão da quantidade de nós: de 1 à p ** (1 / 2).
# https://towardsdatascience.com/frequent-pattern-mining-association-and-correlations-8fa9f80c22ef
# https://lispminer.vse.cz/demonstration/stulong/skupina.html
PARAMETERS = config['PARAMETERS']
DATASET = config["DATASET"]["NAME"]
LABEL = config["DATASET"]["CLASS"]
FEATURES = config["DATASET"]["FEATURES"]
dataset = pd.read_csv(f"./data/raw/{DATASET}.full.csv")

dataset = transform(dataset)

dataset.info()

# FEATURE SELECTION.
# https://scikit-learn.org/stable/modules/feature_selection.html
# fazer para os 10 folds e tirar a media.
# dataset = pd.read_csv(f"./data/raw/{DATASET}.train8.csv")
# features = dataset.drop([LABEL], axis=1)
# labels = dataset[LABEL]
#
# clf = ExtraTreesClassifier(n_estimators=50).fit(features, labels)
# result = permutation_importance(
#     clf, features, labels, n_repeats=50, n_jobs=2
# )
# forest_importances = pd.Series(result.importances_mean, index=features.columns)
#
# # Gráfico
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.show()
#
# print(forest_importances.sort_values(ascending=False))
#
# features_importance_df = pd.DataFrame(
#     features.columns, clf.feature_importances_
# ).reset_index().sort_values(by=["index"], ascending=False)
# features_importance_df.to_csv(
#     f"./data/processed/{DATASET}.feature_importance.csv",
#     index=False
# )
#
# # AMOSTRA.
# sample = dataset.loc[145:150]
# sample.to_csv(f"./data/processed/{DATASET}.sample.csv", index=False)
#
# raw_describe = dataset.describe().T

# ANÁLISE EXPLORATÓRIA.
# Educação.
edu_df = dataset.groupby(["Classe", "Educacao"]).count()

fum_df = dataset.groupby(["Classe", "Fumante"]).count()

ps_df = dataset.groupby(["Classe", "PressaoSanguinea"]).count()

col_df = dataset.groupby(["Classe", "Colesterol"]).count()

imc_df = dataset.groupby(["Classe", "FaixaIMC"]).count()


# CODIFICAÇÃO.
# encoded_dataset = encode(dataset, save=True)
dataset = pd.read_csv(f"./data/processed/{DATASET}/{DATASET}.full.csv")
dataset.info()

# Stratified KFold
# generate_folds(dataset)

# CART
fold = int
for fold in range(1, 11):
    test = pd.read_csv(f"./data/processed/{DATASET}/{DATASET}.test{fold}.csv")
    train = pd.read_csv(f"./data/processed/{DATASET}/{DATASET}.train{fold}.csv")
    train_features = train.drop([LABEL], axis=1)
    train_labels = train[LABEL]
    test_features = test.drop([LABEL], axis=1)
    test_labels = test[LABEL]
    n = len(train)
    J = len(pd.unique(train.Classe))
    p = train_features.shape[1]

    # Fit the model
    cart = tree.DecisionTreeClassifier(
        max_leaf_nodes=8,
        max_features=int(p / 2),
        max_depth=3,
        # class_weight={j: n / (J * train.Classe.value_counts()[j]) for j in range(J)},
        # min_samples_leaf=minleaf,
        # random_state=0,
    )
    cart.fit(train_features, train_labels)

    cart_info = [
        f"DATASET_NAME: {config['DATASET']['NAME']} FOLD: {fold}"
        f"\nCART"
        f"\nCLASS_WEIGHT: {cart.class_weight}"
        f"\nCRITERION: {cart.criterion}"
        f"\nFEATURES: {FEATURES}"
        f"\nN_FEATURES: {cart.tree_.n_features}"
        f"\nN_CLASSES: {cart.tree_.n_classes[0]}"
        f"\nDEPTH: {cart.get_depth()}"
        f"\nN_LEAVES: {cart.get_n_leaves()}"
    ]

    # Predict class labels on training data
    pred_labels_tr = cart.predict(train_features)
    # Predict class labels on a test data
    pred_labels_te = cart.predict(test_features)
    # print(confusion_matrix(test_labels, pred_labels_te))

    # plot confusion matrix
    cmn = normalized_confusion_matrix(test_labels, pred_labels_te)
    # plot_confusion_matrix(cmn, fold)

    cart_info.append(
        f"\n{fold}. Lendo test e train."
        f"\nEvaluation on Test Data\n"
        f"Accuracy Score, {cart.score(test_features, test_labels)}\n"
        f"{classification_report(test_labels, pred_labels_te)}\n"
        f"{confusion_matrix(test_labels, pred_labels_te)}\n"
        # "\n--------------------------------------------------------\n"
        "\nEvaluation on Training Data\n"
        f"Accuracy Score, {cart.score(train_features, train_labels)}\n"
        f"{classification_report(train_labels, pred_labels_tr)}\n"
        f"{confusion_matrix(train_labels, pred_labels_tr)}"
        '\n--------------------------------------------------------\n'
    )
    print(*cart_info)

    # with open(
    #         f"./data/processed/{DATASET}/{DATASET}.CART.MODEL{fold}.pkl",
    #         "wb"
    # ) as f:
    #     dump(cart, f)
    with open(
            f"./data/processed/{DATASET}/{DATASET}.CART.INFO{fold}.txt",
            "w"
    ) as f:
        for item in cart_info:
            f.write(item)


# Random Forest
fold = int
for fold in range(1, 11):
    test = pd.read_csv(f"./data/processed/{DATASET}/{DATASET}.test{fold}.csv")
    train = pd.read_csv(f"./data/processed/{DATASET}/{DATASET}.train{fold}.csv")
    train_features = train.drop([LABEL], axis=1)
    train_labels = train[LABEL]
    test_features = test.drop([LABEL], axis=1)
    test_labels = test[LABEL]
    n = len(train)
    J = len(pd.unique(train.Classe))
    p = train_features.shape[1]

    # with open(
    #         f"./data/processed/{DATASET}/{DATASET}.RF.MODEL4.pkl",
    #         "rb"
    # ) as f:
    #     rf = load(f)

    rf = RandomForestClassifier(
        n_estimators=10,
        max_leaf_nodes=8,
        max_features=int(p / 2),
        max_depth=3,
        class_weight={j: n / (J * train.Classe.value_counts()[j]) for j in range(J)},
    )

    rf.fit(train_features, train_labels)
    print(rf.score(test_features, test_labels))

    # RS
    # parameters = {
    #     "max_leaf_nodes": [*range(10, 30)],
    #     "n_estimators": [*range(1, 20)],
    #     "max_features": [*range(1, rf.n_features_)],
    #     "max_depth": [*range(1, 10)],
    #               }
    # clf = RandomizedSearchCV(rf, parameters, )
    # clf.fit(train_features, train_labels)
    # print(clf.best_params_)

    predictions_test = rf.predict(test.drop([LABEL], axis=1))
    predictions_train = rf.predict(train.drop([LABEL], axis=1))
    print(confusion_matrix(test_labels, predictions_test))
    cmn = normalized_confusion_matrix(test_labels, predictions_test)

    # plot_confusion_matrix(cmn, fold)

    # rf.model_to_txt(index=fold, show=True, save=True)
    # print(confusion_matrix(test[LABEL], predictions_test))
    rf_info = (
        f"\n{fold}. Lendo test e train."
        f"\nEvaluation on Test Data\n"
        f"Accuracy Score, {rf.score(test_features, test_labels)}\n"
        f"{classification_report(test_labels, predictions_test)}\n"
        f"{normalized_confusion_matrix(test_labels, predictions_test)}\n"
        # "\n--------------------------------------------------------\n"
        "\nEvaluation on Training Data\n"
        f"Accuracy Score, {rf.score(train_features, train_labels)}\n"
        f"{classification_report(train_labels, predictions_train)}\n"
        f"{normalized_confusion_matrix(train_labels, predictions_train)}"
        '\n--------------------------------------------------------\n'
    )
    print(rf_info)

    # with open(
    #         f"./data/processed/{DATASET}/4/example/{DATASET}.RF.MODEL{fold}.pkl",
    #         "wb"
    # ) as f:
    #     dump(rf, f)

    with open(
            f"./data/processed/{DATASET}/{DATASET}.RF.INFO{fold}.txt",
            "w"
    ) as f:
        f.write(rf_info)
    model_to_txt(rf, fold, False, True)

