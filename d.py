from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from xgboost import XGBClassifier
import data_handler as dh
def xgbc():

        X_train, X_test, y_train, y_test, scaler = dh.preprocess("bank-full.csv")

        model = XGBClassifier(
                eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        # 1. Save your trained model to a file
        joblib.dump(model, 'xgb.joblib')

        # 2. Load the model back later
        loaded_model = joblib.load('xgb.joblib')

        # acc = accuracy_score(y_test, pred)

        # return f"XGBoost Accuracy: {acc*100:.2f}%"

xgbc()