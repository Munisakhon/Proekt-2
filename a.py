from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
# from xgboost import XGBClassifier
import data_handler as dh
def dtc():

    X_train, X_test, y_train, y_test, scaler = dh.preprocess("bank-full.csv")

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)


# 1. Save your trained model to a file
    joblib.dump(model, 'dtc.joblib')

    # 2. Load the model back later
    loaded_model = joblib.load('dtc.joblib')

    # acc = accuracy_score(y_test, pred)

    # return f"Decision Tree Accuracy: {acc*100:.2f}%"

dtc()