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

x = []
lny = np.log(yy).mean()

lnx1 = np.log(x1).mean()
lnx2 = np.log(x2).mean()

sumlnx1 = np.log(x1).sum()
sumlnx2 = np.log(x2).sum()
sumlnx1pow2 = np.log(x1*x1).sum()
sumlnx2pow2 = np.log(x2*x2).sum()
sumlnx1x2 = np.log(x1+x2).sum()
sumlnx2x1 = np.log(x2+x1).sum()

x = [[1, sumlnx1, sumlnx2],
     [sumlnx1, sumlnx1pow2, sumlnx2x1],
     [sumlnx2, sumlnx1x2, sumlnx2pow2]]

y = [[lny], [lny*lnx1], [lny*lnx2]]

a = (x1**2 * x1**2).sum() - (x1**3).sum()
n = len(yy)
b = a + n + (2 * (x1**3).sum())


beta1 = (((yy*(x1**2)).sum() * b) - (2 * (yy * a).sum()*(x1**2).sum() + (a * n * yy * (x1**2)).sum() + (x1**2).sum() ** 3 - (yy.mean()
                                                                                                                             * (x1).sum() * (x1**2).sum()**2))) / ((((x2 * x1**2).sum() * b) - ((x1**3).sum() * b - (2 * a * n * x1.mean() ** 2 * (x1**2).sum()))))

beta2 = ((((yy*a).sum() - (a * n * yy.mean()) - (a * n * beta1 * (x1.mean() ** 2)) - ((x1**2).sum()
                                                                                      ** 2) + (yy.mean() * (x1).sum() * (x1**2).sum()))) / (((a + n*a) + (2 * (x1**2).sum()))))

beta0 = (((yy.mean() - beta1 * x1.mean() ** 2 - beta2 ** 3)))
betaArray = [[beta0], [beta1], [beta2]]


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
    matrizX = np.insert(matrizX, 0, 1, axis=1)

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

    print(f"{bcolors.HEADER}Ejercicio 6: \n{bcolors.ENDC}")
    print(
        f"La {bcolors.OKBLUE}varianza{bcolors.ENDC} total del modelo es de: {S2_Total} \n")

    # Variaciones

    S2_Matriz = S2_Total * C

    S2_Betas = getDiagonal(S2_Matriz)

    listarBetas(S2_Betas, "Varianza de Beta(s)")

    # Desviaciones


    SqrtS2_Betas = []

    for index, row in enumerate(C):
        for endex, col in enumerate(row):
            if index == endex:
                SqrtS2_Betas.append(math.sqrt(S2_Total) *
                                    math.sqrt(C[index][endex]))

    SqrtS2_Betas = np.array(SqrtS2_Betas)
    print()
    listarBetas(SqrtS2_Betas, "Desviacion Tipica")

    # Prueba Hipotesis
    print()
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


realizarPrueba()
