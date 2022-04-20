from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pickle import load, dump
from pandas import DataFrame
from os import getcwd
from datetime import datetime

TODAY = datetime.today()
PATH = getcwd()
MODEL_DATE = "20211230"


def process_pca(dataframe,
                load_model: bool = True,
                explained_percentage: float = .8,
                save: bool = False,
                ):
    """
    Function to select linear combinations of variables to "replace" the original ones
    without much loss of information.
    Parameters
    ----------
    dataframe
    load_model
    explained_percentage
    save

    Returns
    -------

    """
    print(f"Processing PCA with minimum of {explained_percentage} explained percentage of"
          f" variance. Loading Model: {load_model}")
    # Completando os NAs.
    x_train = dataframe.copy()
    if x_train.isnull().values.any():
        x_train.fillna(-99, inplace=True)
    # columns = BIC_COLUMNS.copy()
    # columns.extend(COLUMNS)
    # x_train = x_train[columns]
    # Padronizando os valores.
    x = StandardScaler().fit_transform(x_train)
    # Quantidade de colunas/variáveis.
    n_columns = len(x[1, :])
    # Carregando/criando o modelo.
    if load_model:
        try:
            print("Loading model...")
            model = load(open(f"{PATH}/model/{MODEL_DATE}_pca.pkl", "rb"))
        except Exception as e:
            raise Exception(f"Erro ao abrir o modelo: {e}")
    else:
        model = PCA(
            n_components=n_columns
        )

    principal_dataframe = DataFrame(
        data=model.fit_transform(x),
        columns=[f"PC{x}" for x in range(0, n_columns)]
    )
    #
    q_components = 0
    explained_variance_ratio = model.explained_variance_ratio_.cumsum()
    # print(f"Variância acumulada: {explained_variance_ratio}")
    for cum_var in explained_variance_ratio:
        if cum_var >= explained_percentage:
            q_components = len(
                explained_variance_ratio[explained_variance_ratio < explained_percentage]
            )
            print(f"Components: {q_components}. \n "
                  f"Explained variance ration: {cum_var:.2f}%.")
            break
    # Quantidade de componentes principais escolhidos (70% ~ 80%).
    dataframe_pca = principal_dataframe.iloc[:, :q_components]

    if save:
        filename = f"{TODAY.strftime('%Y%m%d')}_pca.pkl"
        print(f"Saving {filename}...")
        dump(model, open(
            f"{PATH}/model/{filename}",
            "wb",
        ))
    return dataframe_pca
