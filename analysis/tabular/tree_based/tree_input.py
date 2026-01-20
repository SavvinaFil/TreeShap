import pandas as pd
import pickle


def load_tree(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_dataset(choice, feature_names=None, path_override=None):
    if path_override:
        path = path_override
    else:
        raise ValueError("Path must be provided via config")

    if choice == 2:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if feature_names is None:
        feature_names = list(df.columns)

    return df[feature_names]