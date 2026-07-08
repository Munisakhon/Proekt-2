import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(path):

    df = pd.read_csv(path, sep=";")

    df["housing"] = df["housing"].map({
        "yes":1,
        "no":0
    })

    df["loan"] = df["loan"].map({
        "yes":1,
        "no":0
    })

    df["y"] = df["y"].map({
        "yes":1,
        "no":0
    })

    df = df[
        [
            "age",
            "balance",
            "duration",
            "campaign",
            "previous",
            "housing",
            "loan",
            "y"
        ]
    ]

    X = df.drop("y", axis=1)
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )