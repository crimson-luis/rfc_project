import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class GaussianModel:
    def __init__(self, x: list, y: list, data: pd.DataFrame, estimators: int = 100):
        self.data = data
        self.clf = RandomForestClassifier(n_estimators=estimators)
        self.x = x
        self.y = y
        self.accuracy = None

    def train(self, test_size: float = 0.3):
        (
            explanatory_train,
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
