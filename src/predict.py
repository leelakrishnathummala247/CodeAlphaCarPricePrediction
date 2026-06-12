import pandas as pd
import joblib

def predict_price():

    model = joblib.load(
        "models/carpricemodel.pkl"
    )

    features = joblib.load(
        "models/features.pkl"
    )

    sample = pd.DataFrame(
        [[2018,50000,1,0,0]],
        columns=features[:5]
    )

    prediction = model.predict(sample)

    print(
        "Predicted Car Price :",
        prediction[0]
    )