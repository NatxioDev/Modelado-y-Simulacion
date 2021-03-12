import pandas as pd
import numpy as np
import math


def newIdent(n):
    matriz = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matriz[i][j] = 1

    return matriz


def toTransForce(matriz):

    trans = (np.zeros((len(matriz.columns), len(matriz))))

    for i in range(len(matriz)):
        for j in range(len(matriz.columns)):

            trans[j][i] = matriz.iloc[i, j]

    return trans


def getMatriz(matriz):

    mat = (np.zeros((len(matriz), len(matriz.columns))))

    for i in range(len(matriz)):
        for j in range(len(matriz.columns)):

            mat[i][j] = matriz.iloc[i, j]

    return mat


def getInversa(matriz):

    identidad = newIdent(len(matriz))

    return identidad


def getDiagonal(matriz):

    diagonal = []

    for index, row in enumerate(matriz, start=0):
        for endex, column in enumerate(row, start=0):
            if index == endex:
                diagonal.append(column)

    return diagonal


def listarBetas(array, msg):

    print(msg)

    for index, Beta in enumerate(array, start=1):

        print(f'Beta {index}: {Beta}')


fileData = pd.read_csv("./data.csv")

dataY = fileData.iloc[:, 0:1]
dataX = fileData.iloc[:, 1:3]


matrizX = getMatriz(dataX)
matrizX = np.insert(matrizX, 0, 1, axis=1)


matrizXTranspuesta = np.transpose(matrizX)



matrizY = getMatriz(dataY)
matrizYTranspuesta = toTransForce(dataY)

MT_X_M = np.matmul(matrizXTranspuesta, matrizX)

C = np.linalg.inv(MT_X_M)



aux = np.matmul(C, matrizXTranspuesta)


BetaS = (np.matmul(aux, matrizY))



YS = np.matmul(matrizX, BetaS)

print(YS)

SumY_YS = (((matrizY - YS) * (matrizY - YS)).sum())
div = (len(dataY) - len(dataX.columns))

S2_Total = SumY_YS / div

print("La varianza total del modelo es de:", S2_Total)

S2_Matriz = S2_Total * C

S2_Betas = getDiagonal(S2_Matriz)

listarBetas(S2_Betas, "Varianza")

#Desviaciones
SqrtS2_Matriz = math.sqrt(S2_Total) * C

SqrtS2_Betas = getDiagonal(SqrtS2_Matriz)

listarBetas(SqrtS2_Betas, "Desviacion Tipica")



