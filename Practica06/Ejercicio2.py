import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("./datac.xlsx")
Q = 5

T = len(data)
i = 1
Y = data.iloc[Q:, :]
lenData = len(data)

while len(Y) > Q:
    print("Iteracion :", i)
    Y = data.iloc[i:, :]
    X = data.iloc[:lenData-i, :]
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)
    Y = np.array(Y)
    Y = np.transpose(Y)
    COVXY = ((X - X.mean()).sum() * (Y - Y.mean()).sum()) / lenData
    print("Covarianza", COVXY)
    print(Y * COVXY)

    if COVXY == 0:
        break
    Y = data.iloc[i:, :]
    X = data.iloc[:lenData-i, :]
    i = i + 1

