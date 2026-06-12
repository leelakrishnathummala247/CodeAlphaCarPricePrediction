from src.loaddataset import load_dataset
from src.trainmodel import train_model
from src.evaluatemodel import evaluate_model
from src.predict import predict_price

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = load_dataset()

os.makedirs(
    "outputs",
    exist_ok=True
)

numeric_data = data.select_dtypes(
    include="number"
)

plt.figure(
    figsize=(8,6)
)

sns.heatmap(
    numeric_data.corr(),
    annot=True
)

plt.savefig(
    "outputs/correlationheatmap.png"
)

plt.close()

train_model()

evaluate_model()

predict_price()