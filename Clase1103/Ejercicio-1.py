import numpy as np
import math as m


y = [[2], [4], [6], [8], [10], [12]]
x = [[1], [2], [3], [4], [5], [6]]

matrizX = np.array(x)
matrizY = np.array(y)
matrizX = np.insert(matrizX, 0, 1, axis=1)
matrizXTranspuesta = matrizX.transpose()

MT_X_M = np.matmul(matrizXTranspuesta, matrizX)

C = np.linalg.inv(MT_X_M)
aux = np.matmul(C, matrizXTranspuesta)

BetaS = (np.matmul(aux, matrizY))

YS = np.matmul(matrizX, BetaS)


beta1 = ((matrizY * matrizX).sum() - matrizY.mean() * (matrizX).sum()
         ) / ((matrizX**2).sum() - matrizX.mean() * (matrizX).sum())
beta0 = matrizY.mean() - (beta1 * matrizX.mean())
betaArray = [[beta0], [beta1]]



SEC = (YS - YS.mean()).sum() ** 2
SRC = (matrizY-YS).sum() ** 2
STC = SEC + SRC

print("SEC: ", SEC)
print("SRC: ", SRC)
print("STC: ", STC)

GLN = 1  # Numerador
GLD = 5  # Dividendo



if (beta0 != 0 and beta1 != 0):
    print("Se acepta la hipotesis nula")

    F = ((SEC/GLN) / (SRC / GLD))

    FF = 10.007

    if(F > FF):
        print("Se rechaza la hipotesis nula, se acepta la hipotesis alterna")
    else:
        print("Se acepta la hipotesis nula y se rechaza la hipotesis alterna")
else:

    print("Se rechaza la hipotesis nula")