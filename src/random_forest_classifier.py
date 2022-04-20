import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


class GaussianModel:
    def __init__(self, x: list, y: list, data: pd.DataFrame, estimators: int = 100):
        self.data = data
        self.clf = RandomForestClassifier(
            n_jobs=1,
            verbose=0,
            bootstrap=True,
            max_depth=3,
            oob_score=False,
            criterion='gini',
            warm_start=False,
            class_weight=None,
            random_state=None,
            min_samples_leaf=1,
            max_features='auto',
            max_leaf_nodes=None,
            min_samples_split=2,
            n_estimators=estimators,
            min_impurity_split=None,
            min_impurity_decrease=0.0,
            min_weight_fraction_leaf=0.0,
        )
        self.x = x
        self.y = y
        self.accuracy = None

    def train(self, test_size: float = 0.3):
        (explanatory_train,
         explanatory_test,
         response_train,
         response_test,
         ) = train_test_split(
            self.data[self.x], self.data[self.y], test_size=test_size
        )
        self.clf.fit(explanatory_train, response_train)
        predicted_response = self.clf.predict(explanatory_test)
        self.accuracy = metrics.accuracy_score(response_test, predicted_response)

    def classify(self, info: list):
        assert len(info) == len(self.x), "Dimensions don't match."
        return self.clf.predict([info])

    def feature_score(self, plot: bool = False):
        feature_importance = pd.Series(
            data=self.clf.feature_importances_,
            index=self.x,
        ).sort_values(ascending=False)
        if plot:
            sns.barplot(
                x=feature_importance,
                y=feature_importance.index
            )
            plt.xlabel("Feature Importance Score")
            plt.ylabel("Features")
            plt.title("Visualizing Important Features")
            plt.legend()
            plt.show()
        return feature_importance
