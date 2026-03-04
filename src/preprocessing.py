import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def select_features(df):

    features = [
        "glucose_fasting",
        "glucose_postprandial",
        "hba1c",
        "insulin_level",
        "bmi"
    ]

    X = df[features]

    return X