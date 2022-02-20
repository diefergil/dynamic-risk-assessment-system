import pickle
import pandas as pd
from typing import Tuple
import sklearn

def load_data(data_path: str) -> Tuple[pd.DataFrame, list]:
    """
    Loads dataset, drop unused columns and return `x` dataframe and `y` labels.
    """
    df = pd.read_csv(data_path).drop("corporation", axis=1)
    y = df.exited.values
    X = df.drop("exited", axis=1)

    return X, y


def load_model(model_path: str):
    """
    Loads output model from training step, according
    to config.json file. Then returns that model.
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model
