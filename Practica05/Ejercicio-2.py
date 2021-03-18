import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

plt.close("all")

fileData = pd.read_csv("./data.csv")

# Obtencion de columnas
yy = fileData.iloc[:, 0]  # promedio
x1 = fileData.iloc[:, 1]  # edad
x2 = fileData.iloc[:, 2]  # materias
x3 = fileData.iloc[:, 3]  # horasEstudio
x4 = fileData.iloc[:, 4]  # recreo
x5 = fileData.iloc[:, 5]  # amigos


beta2 = (((yy * x2).sum() * (x1**2).sum()) - ((yy * x1).sum() *
                                              (x1 * x2).sum())) / (((x2**2).sum() * (x1).sum()) - ((x1*x2).sum())**2)
beta1 = ((yy*x1).sum() * (x2**2).sum() - (yy * x2).sum() * (x2*x1).sum()
         ) / (x1**2).sum() * ((x1**2).sum()*(x1*x2).sum() * (x2*x1).sum())

betaArray = [[beta1], [beta2]]


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

        print(f'{bcolors.BOLD}Beta {index}{bcolors.ENDC}: {Beta}')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAGENTA = '\u001b[35m'


def realizarPrueba():

    fileData = pd.read_csv("./data.csv")

    dataY = fileData.iloc[:, 0:1]
    dataX = fileData.iloc[:, 1:3]

    matrizX = getMatriz(dataX)
    # matrizX = np.insert(matrizX, 0, 1, axis=1)

    matrizXTranspuesta = np.transpose(matrizX)

    matrizY = getMatriz(dataY)
    matrizYTranspuesta = toTransForce(dataY)

    MT_X_M = np.matmul(matrizXTranspuesta, matrizX)

    C = np.linalg.inv(MT_X_M)

    aux = np.matmul(C, matrizXTranspuesta)

    BetaS = (np.matmul(aux, matrizY))

    YS = np.matmul(matrizX, BetaS)

    SumY_YS = (((matrizY - YS) * (matrizY - YS)).sum())
    div = (len(dataY) - len(dataX.columns))

    S2_Total = SumY_YS / div

    print(f"{bcolors.HEADER}Ejercicio 2: \n{bcolors.ENDC}")
    print(
        f"La {bcolors.OKBLUE}varianza{bcolors.ENDC} total del modelo es de: {S2_Total} \n")

    # Variaciones

    S2_Matriz = S2_Total * C

    S2_Betas = getDiagonal(S2_Matriz)

    listarBetas(S2_Betas, "Varianza de Beta(s)")

    SqrtS2_Betas = []

    for index, row in enumerate(C):
        for endex, col in enumerate(row):
            if index == endex:
                SqrtS2_Betas.append(math.sqrt(S2_Total) *
                                    math.sqrt(col))

    SqrtS2_Betas = np.array(SqrtS2_Betas)

    SqrtS2_Betas = np.array(SqrtS2_Betas)
    print()
    listarBetas(SqrtS2_Betas, "Desviacion Tipica de Beta(s)")

    # Prueba Hipotesis
    t = float(input(f"{bcolors.OKBLUE}Ingrese el valor de t: {bcolors.ENDC}"))

    T1 = (t * SqrtS2_Betas.mean()) + BetaS.mean()
    T2 = BetaS.mean() - (t * SqrtS2_Betas.mean())

    print()
    print(f"{bcolors.OKCYAN}LIC ->{bcolors.ENDC} {T2}")
    print(f"{bcolors.OKCYAN}LSC ->{bcolors.ENDC} {T1}")
    print(f"\n{T2} {bcolors.MAGENTA}  >  \u03B2  >  {bcolors.ENDC} {T1}\n")

    for beta in BetaS:
        if beta == 0:
            print(
                f"{bcolors.FAIL}Se elimina debido a que el coeficiente {beta} es igual a cero{bcolors.ENDC}")
        else:
            print(
                f"{bcolors.OKGREEN}No se elimina, debido a que el coeficiente {beta} es diferente de Cero{bcolors.ENDC}")

    print()

    for index, beta in enumerate(BetaS):
        ti = beta/SqrtS2_Betas[index]
        if ti > t:
            print(
                f"{bcolors.WARNING}Se rechaza la hipotesis nula:{bcolors.ENDC} {ti} > {t}")
        else:
            print(
                f"{bcolors.OKCYAN}Se acepta la hipotesis nula:{bcolors.ENDC} {ti} < {t}")

    SEC = (YS - YS.mean()).sum() ** 2
    SRC = (np.array(yy)-YS).sum() ** 2
    STC = SEC + SRC

    GLN = len(BetaS)
    GLD = len(yy) - len(BetaS)

    print(f"{bcolors.OKCYAN}SEC: {bcolors.ENDC}", SEC)
    print(f"{bcolors.OKCYAN}SRC: {bcolors.ENDC}", SRC)
    print(f"{bcolors.OKCYAN}STC: {bcolors.ENDC}", STC)
    print()
    print(f"{bcolors.MAGENTA}GLN: {bcolors.ENDC}", GLN)
    print(f"{bcolors.MAGENTA}GLD: {bcolors.ENDC}", GLD)
    print()

    flag = True
    for veta in betaArray:
        if veta == 0:
            flag = False

    if (flag != False):

        F = ((SEC/GLN) / (SRC / GLD))

        FF = 4.106
        print(f"{bcolors.OKCYAN}El valor de F -> {bcolors.ENDC}{F}")
        if(F > FF):
            print(
                f"{bcolors.WARNING}Se rechaza la hipotesis nula, se acepta la hipotesis alterna{bcolors.ENDC}")
        else:
            print(
                f"{bcolors.OKGREEN}Se acepta la hipotesis nula y se rechaza la hipotesis alterna{bcolors.ENDC}")
    else:

        print(f"{bcolors.FAIL}Se rechaza la hipotesis nula{bcolors.ENDC}")


realizarPrueba()
