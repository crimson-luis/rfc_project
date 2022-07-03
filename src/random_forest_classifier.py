from sklearn.ensemble import RandomForestClassifier
import numpy as np
import yaml

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

DATASET = config['DATASET']['NAME']


class RFClassifier(RandomForestClassifier):
    def __init__(self):
        super().__init__()
        # self.parameters = config['PARAMETERS']
        # self.max_leaf_nodes = self.parameters['MAX_LEAF_NODES']
        # self.n_estimators = self.parameters['N_ESTIMATORS']
        # self.max_features = self.parameters['MAX_FEATURES']
        # self.max_depth = self.parameters['MAX_DEPTH']

    def model_to_txt(self, index, show: bool = True, save: bool = False):
        # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
        forest = self.estimators_
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
                    f"./data/processed/{DATASET}/{DATASET}.RF{index}.txt",
                    "w"
            ) as f:
                for item in model_info:
                    f.write(item)
