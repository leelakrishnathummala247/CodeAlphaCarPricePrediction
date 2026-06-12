import pandas as pd

def load_dataset():

    data = pd.read_csv("dataset/cardata.csv")

    print("Dataset Loaded Successfully")

    print(data.head())

    return data