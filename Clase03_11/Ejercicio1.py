import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("./datac.xlsx")
Q = 15

lambdaArray = np.array([])
lambdaArray2 = np.array([])
X = np.array(data)
X2 = np.array(data)
Y = np.array(data)
Y2 = np.array(data)

T = len(data)
X = np.delete(X, len(X)-1, axis=0)
Y = np.delete(Y, 0, axis=0)
X2 = np.delete(X2, len(X2)-1, axis=0)
Y2 = np.delete(Y2, 0, axis=0)
YT = 0
YT2 = 0


for i in range(1, Q+1, 1):
    print("Iteracion :", i)
    COVXY = ((X - X.mean()).sum() * (Y - Y.mean()).sum()) / len(Y)
    print("Covarianza", COVXY)


    XT = np.transpose(X)
    C = np.linalg.inv(np.matmul(XT,X))
    XT_x_Y = np.matmul(XT,Y)
    BetaS = np.matmul(C,XT_x_Y)
    lambdaArray = np.append(lambdaArray, BetaS)
    YT = YT + (BetaS * X[0])


    X = np.delete(X, len(X)-1, axis=0)
    Y = np.delete(Y, 0, axis=0)

t = []
for e in range(0, len(lambdaArray), 1):
    t.append(i+1)

plt.plot(t, lambdaArray)
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
