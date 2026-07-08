from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import data_handler as dh


def models():

    X_train, X_test, y_train, y_test, scaler = dh.preprocess(
        "bank-full.csv"
    )

    model1 = LogisticRegression()
    model1.fit(X_train, y_train)

    model2 = DecisionTreeClassifier()
    model2.fit(X_train, y_train)


    model3 = XGBClassifier(
        eval_metric="logloss"
    )
    model3.fit(X_train, y_train)

    return (
        model1,
        model2,
        model3
    )