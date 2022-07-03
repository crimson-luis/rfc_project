from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from yaml import safe_load
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

with open("./config.yaml", "r") as file:
    config = safe_load(file)

DATASET = config["DATASET"]["NAME"]
LABEL = config["DATASET"]["CLASS"]
FEATURES = config["DATASET"]["FEATURES"]
OUT_DIR = f"./data/processed/{DATASET}"
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


def generate_folds(df):
    y = df[LABEL]
    n = df.shape[0]
    skf = StratifiedKFold(n_splits=config["DATASET"]["N_SPLITS"])
    i = 1
    for train, test in skf.split(np.zeros(n), y):
        print(f"Salvando: {DATASET}.train{i}.csv e {DATASET}.test{i}.csv...")
        df.iloc[train.astype(int)].to_csv(
            os.path.join(OUT_DIR, f"{DATASET}.train{i}.csv"),
            index=False,
        )
        df.iloc[test.astype(int)].to_csv(
            os.path.join(OUT_DIR, f"{DATASET}.test{i}.csv"),
            index=False,
        )
        i += 1


def encode(dataset, ignore: list = None, encode_type: str = "F", save: bool = False):
    # Classes are always indexed from 0 to K-1.
    drop = [LABEL, *ignore] if ignore else [LABEL]
    x = dataset.drop(drop, axis=1)
    y = dataset[LABEL]
    if encode_type == "OH":
        enc = OneHotEncoder().fit(x)
        x = pd.DataFrame(
            data=enc.transform(x).toarray(),
            columns=enc.get_feature_names(np.array(list(x.columns)))
        )
    elif encode_type == "F":
        try:
            x["Urine"] = np.where(x.Urine == "normal", "1", x.Urine)
            x["BMIRange"] = np.where(x.BMIRange == "normal", "2", x.BMIRange)
        except AttributeError:
            pass
        replace_str = {
            # Urine
            # "normal": "1",
            "albumeno positivo": "2",
            "acucar positivo": "3",
            # Colesterol
            "limitrofe": "1",
            "desejavel": "2",
            # BloodPressure
            "normal": "1",
            "normal/alto": "2",
            "alto": "3",
            # BMIRange
            "baixo peso": "1",
            # "normal": "2",
            "excesso de peso": "3",
            "obeso": "4",
            "obeso morbido": "5",
            # WeightRange
            "< 60": "1",
            "100 - 109": "> 2",  # "> 2" > "> 119"
            "110 - 119": "> 3",  # "> 2" > "> 119"
            # HeightRange
            "< 1.60": "1",
            # DailyLiquorCons, DailyWineCons
            "nao bebe alcool": "1",
            # DailyLiquorCons
            "nao bebe licor": "2",
            "ate 100cc": "3",
            "mais de 100cc": "4",
            # DailyWineCons
            "nao bebe vinho": "2",
            "ate meio litro": "3",
            "mais de meio litro": "4",
            # DailyBeerCons
            "nao bebe cerveja": "2",
            "ate um litro": "3",
            "mais de um litro": "4",
            # ExSmoker
            "nunca fumou": "1",
            "sim, menos de um ano": "2",
            "continua fumando": "3",
            "sim, mais de um ano": "4",
            # SmokingDuration/Smoking
            "nao fumante": "1",
            # SmokingDuration
            "ate 5 anos": "2",
            "6-10 anos": "3",
            "11-20 anos": "4",
            "21 anos ou mais": "5",
            # Smoking
            "1-4 cig/dia": "2",
            "5-14 cig/dia": "3",
            "15-20 cig/dia": "4",
            "21 ou mais cig/dia": "5",
            # JobTranspDuration
            "por volta de 1/2 hora": "1",
            "por volta de 1 hora": "2",
            "por volta de 2 horas": "3",
            "mais de 2 horas": "4",
            # Transport
            "a pe": "1",
            "bicicleta": "2",
            "transporte publico": "3",
            "carro": "4",
            # Transport
            "otima atividade": "1",
            "atividade moderada": "2",
            # "senta-se principalmente": "3",
            # PhysActInJob
            "carrega carga pesada": "1",
            "anda principalmente": "2",
            "principalmente em pe": "3",
            # "senta-se principalmente": "3",
            # Educação
            "apprentice school": "1",
            "basic school": "2",
            "ensino medio": "3",
            "universidade": "4",
        }
        x = x.replace(replace_str)
        x = x.apply(lambda k: pd.factorize(k, sort=True)[0] + 1)
    concat = [x, dataset[ignore], y] if ignore else [x, y]
    dataset = pd.concat(concat, axis=1)
    dataset[LABEL] = pd.factorize(y, sort=True)[0]
    dataset = dataset.apply(lambda h: pd.to_numeric(h))
    if save:
        print(f"Saving: {DATASET}.full.csv...")
        dataset.to_csv(os.path.join(OUT_DIR, f"{DATASET}.full.csv"), index=False)

    return dataset


def transform(dataset):
    # Samples containing missing values have been eliminated.
    print("Substituindo NS e ? por NaN.")
    dataset = dataset.replace({"NS": np.NAN, "?": np.NAN})

    if DATASET == "STULONG":
        # A: Normal Group
        # B: Risk Group
        # C: Pathologic Group
        dataset.drop(["GroupCode"], axis=1, inplace=True)
        # dataset[LABEL] = dataset[LABEL].replace("B", "A")
        # dataset.DobraCutaneaSube = pd.to_numeric(dataset.DobraCutaneaSube, errors="coerce")
        # dataset.DobraCutaneaTric = pd.to_numeric(dataset.DobraCutaneaTric, errors="coerce")
        # dataset["DobraCutanea"] = dataset.DobraCutaneaSube + dataset.DobraCutaneaTric
        # dataset.drop(columns=["DobraCutaneaSube", "DobraCutaneaTric"], inplace=True)
        # dataset.drop(columns=["Triglicerideos"], inplace=True)
        # dataset["DobraCutanea"] = np.where(
        #     dataset.DobraCutanea < 20,
        #     "8 - 20",  # 8 - 20
        #     np.where(
        #         dataset.DobraCutanea < 30,
        #         "21 - 30",  # 21 - 30
        #         np.where(
        #             dataset.DobraCutanea < 40,
        #             "31 - 40",  # 31 - 40
        #             "> 40"  # > 40
        #         )
        #     )
        # )

    elif DATASET == "THYROID":
        dataset[LABEL] = dataset[LABEL + "[record_identification]"].str[:1]
        dataset["[record_identification]"] = dataset[LABEL + "[record_identification]"].str[1:]
        dataset[LABEL] = np.where(
            np.isin(dataset[LABEL], ("A", "B", "C", "D")),
            "A",  # Hyperthyroid conditions
            np.where(
                np.isin(dataset[LABEL], ("E", "F", "G", "H")),
                "B",  # Hypothyroid conditions
                "-",  # no condition requiring comment
            )
        )

    # SELEÇÃO DE FEATURES.
    print("Selecionando apenas atributos escolhidos")
    dataset = dataset[FEATURES]
    print("Removendo os casos com informações faltantes.")
    dataset = dataset.dropna().reset_index(drop=True)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    return dataset


# PX
def model_to_txt(model, index, show: bool = True, save: bool = False):
    # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    forest = model.estimators_
    model_info = list()
    model_info.append(
        f"DATASET_NAME: {config['DATASET']['NAME']}.train{index}.csv"
        f"\nENSEMBLE: RF"
        f"\nNB_TREES: {len(forest)}"
        f"\nNB_FEATURES: {forest[0].tree_.n_features}"
        f"\nNB_CLASSES: {forest[0].tree_.n_classes[0]}"
        f"\nMAX_TREE_DEPTH: {forest[0].tree_.max_depth}"
        "\nFormat: node / node type (LN - leave node, IN - internal node) "
        "left child / right child / feature / threshold / node_depth / "
        "majority class (starts with index 0)"
    )
    for tree_idx, est in enumerate(forest):
        tree = est.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        model_info.append(f"\n\n[TREE {tree_idx}]\nNB_NODES: {n_nodes}")
        # Calculating depth.
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node;
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        for i in range(n_nodes):
            class_idx = np.argmax(tree.value[i][0])
            if is_leaves[i]:
                model_info.append(f"\n{i} LN -1 -1 -1 -1 {node_depth[i]} {class_idx}")
            else:
                model_info.append(
                    f"\n{i} IN {children_left[i]} {children_right[i]} "
                    f"{tree.feature[i]} {tree.threshold[i]} {node_depth[i]} -1"
                )
    model_info.append("\n\n")
    if show:
        print(*model_info)
    if save:
        with open(
                os.path.join(OUT_DIR, f"{DATASET}.RF{index}.txt"),
                "w"
        ) as f:
            for item in model_info:
                f.write(item)


def normalized_confusion_matrix(label, pred):
    cm = confusion_matrix(label, pred)
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


def plot_confusion_matrix(cmn, fold, show: bool = False, save: bool = True):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cmn, vmin=0, vmax=1, annot=True, fmt='.2f', xticklabels=["A", "B", "C"],
                yticklabels=["A", "B", "C"])
    plt.ylabel('Real')
    plt.xlabel('Predito')
    if show:
        plt.show(block=False)
    if save:
        plt.savefig(os.path.join(OUT_DIR, f"{DATASET}.RF.NCM{fold}.png"))
