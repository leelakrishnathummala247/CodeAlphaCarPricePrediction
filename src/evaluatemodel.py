import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def evaluate_model():

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

    model = joblib.load(
        "models/carpricemodel.pkl"
    )

    predictions = model.predict(X_test)

    score = r2_score(
        y_test,
        predictions
    )

    os.makedirs("outputs", exist_ok=True)

    with open(
        "outputs/predictionresults.txt",
        "w"
    ) as file:

        file.write(
            f"R2 Score : {score}"
        )

    print("R2 Score :", score)