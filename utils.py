import pickle
import pandas as pd
from typing import Tuple
import sklearn
import pandas as pd

def splitting_data(df: pd.DataFrame):
    y = df.exited.values
    X = df.drop(["corporation", "exited"], axis=1)
    
    return X, y
    

def load_data(data_path: str) -> Tuple[pd.DataFrame, list]:
    """
    Loads dataset, drop unused columns and return `x` dataframe and `y` labels.
    """
    df = pd.read_csv(data_path)
    X, y = splitting_data(df)
    return X, y


def load_model(model_path: str):
    """
    Loads output model from training step, according
    to config.json file. Then returns that model.
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model
