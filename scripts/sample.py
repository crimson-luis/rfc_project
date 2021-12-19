import pandas as pd
import numpy as np


class Hajek:
    def __init__(self, database: pd.DataFrame):
        self.database = database
        self.N = len(self.database)

    def select_sample(self, n: int = 32):
        self.database["uniform_values"] = np.random.uniform(low=0, high=1, size=self.N)
        self.database.sort_values(["uniform_values"], inplace=True, ignore_index=True)
        return self.database.loc[range(0, n), :]
