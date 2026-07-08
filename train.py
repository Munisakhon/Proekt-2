from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

import data_handler as dh


# import streamlit as st

class Models:

    def lr():

        # X_train, X_test, y_train, y_test, scaler = dh.preprocess("bank-full.csv")

        # model = LogisticRegression()

        # model.fit(X_train, y_train)

        # pred = model.predict(X_test)

        # 1. Save your trained model to a file
        # joblib.dump(model, 'my_model.joblib')

        # 2. Load the model back later
        model = joblib.load('lr.joblib')
        pred = model.predict(X_test)

        # acc = accuracy_score(y_test, pred)

        # return f"Logistic Regression Accuracy: {acc*100:.2f}%"

    def dtc():

    #     X_train, X_test, y_train, y_test, scaler = dh.preprocess("bank-full.csv")

    #     model = DecisionTreeClassifier()

    #     model.fit(X_train, y_train)

    #     pred = model.predict(X_test)


    # # 1. Save your trained model to a file
    #     joblib.dump(model, 'my_model.joblib')

        # 2. Load the model back later
        model = joblib.load('dtc.joblib')
        pred = model.predict(X_test)
        # acc = accuracy_score(y_test, pred)

        # return f"Decision Tree Accuracy: {acc*100:.2f}%"


    def xgbc():

        # X_train, X_test, y_train, y_test, scaler = dh.preprocess("bank-full.csv")

        # model = XGBClassifier(
        #     eval_metric="logloss"
        # )

        # model.fit(X_train, y_train)

        # pred = model.predict(X_test)

        # 1. Save your trained model to a file
        # joblib.dump(model, 'xgb.joblib')

        # 2. Load the model back later
        model = joblib.load('rfc.joblib')
        pred = model.predict(X_test)

        # acc = accuracy_score(y_test, pred)

        # return f"XGBoost Accuracy: {acc*100:.2f}%"
    
def get_all_scores():

    X_train, X_test, y_train, y_test, scaler = dh.preprocess(
        "bank-full.csv"
    )

    scores = {}

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    scores["Logistic Regression"] = (
        accuracy_score(y_test, lr.predict(X_test))*100
    )

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    scores["DecisionTreeClassifier"] = (
        accuracy_score(y_test, dt.predict(X_test))*100
    )


    xgb = XGBClassifier(
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    scores["XGBClassifier"] = (
        accuracy_score(y_test, xgb.predict(X_test))*100
    )

    return scores