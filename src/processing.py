import pandas as pd


def factorize_data(dataset: pd.DataFrame, show: bool = True):
    columns = dataset.columns
    print(f"{columns}\n")
    factorized_dataframe = pd.DataFrame()
    for column in columns:
        factorized_column = pd.factorize(dataset[column], sort=True)[0]
        factorized_dataframe[f"{column}_f"] = factorized_column
        check_unique = pd.DataFrame(
            [factorized_column, dataset[column]],
            index=[f"{column}_f", f"{column}"]
        )
        if show:
            print(check_unique.T.drop_duplicates(
                subset=[f"{column}_f"]
            ).sort_values(by=[column]))
    return factorized_dataframe

