import pandas as pd
from sklearn import datasets
from src.random_forest_classifier import GaussianModel
from src import __version__
# from scripts.sample import Hajek


def test_version():
    assert __version__ == '0.1.0'


def run_rf_test():
    model = GaussianModel(
        x=['sepal length',
           'sepal width',
           'petal length',
           'petal width'],
        y=['species'],
        data=build_test_df(),
    )
    print(model.accuracy)
    model.train(test_size=0.3)
    predicted = model.classify(info=[3, 5, 4, 2])
    print(f"Predicted value for species: {predicted}")


def build_test_df():
    iris = datasets.load_iris()
    return pd.DataFrame(
        {'sepal length': iris.data[:, 0],
         'sepal width': iris.data[:, 1],
         'petal length': iris.data[:, 2],
         'petal width': iris.data[:, 3],
         'species': iris.target}
    )
