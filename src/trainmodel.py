import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model():

    data = pd.read_csv("dataset/cardata.csv")

    data = pd.get_dummies(data, drop_first=True)

    X = data.drop("Selling_Price", axis=1)

    y = data["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/carpricemodel.pkl"
    )

    joblib.dump(
        X.columns,
        "models/features.pkl"
    )

    print("Model Trained Successfully")

    return X_test, y_test