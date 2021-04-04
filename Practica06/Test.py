import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("./datac.xlsx")
Q = 5
lambdaArray = np.array([])
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

    if(COVXY == 0):
        break

    Y = data.iloc[i:, :]
    X = data.iloc[:lenData-i, :]
    i = i + 1


# Y = np.array(DataY)

# X = np.array(DataX)
# X = np.insert(X, 0, 1, axis=1)

# print("Data total", data)
# print("Data Y", DataY)
# print("Data X", DataX)
# print("Y: ", Y)
# print("X: ", X)


# YTranspuesta = np.transpose(Y)
# XTranspuesta = np.transpose(X)

# XT_x_X = np.matmul(XTranspuesta, X)

# C = np.linalg.inv(XT_x_X)

# aux = np.matmul(C, XTranspuesta)

# BetaS = (np.matmul(aux, Y))

# YS = np.matmul(X, BetaS)

# print("Betas: ", BetaS)
# print("Y estimado: ", YS)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.close("all")
data = pd.read_excel("../totalData.xlsx")
Q = 1000

