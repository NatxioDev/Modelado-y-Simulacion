import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def getMano(element):
    if element == 1:
        return "Q"
    numberDic = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
    }
    element = str(round(element, 5))
    element = element.replace(".", "")
    for index, caracter in enumerate(element, start=0):
        if index != 0:
            numberDic[caracter] += 1

    return getRepeticiones(numberDic)


def getRepeticiones(numberDic):
    c1 = c2 = c3 = c4 = c5 = 0

    for item in numberDic:
        cant = numberDic[item]
        if cant == 1:
            c1 += 1
        if cant == 2:
            c2 += 1
        if cant == 3:
            c3 += 1
        if cant == 4:
            c4 += 1
        if cant == 5:
            c5 += 1

    if c5 == 5:
        return "Q"
    elif c4 == 4:
        return "P"
    elif c3 == 1 and 1 == c2:
        return "TP"
    elif c3 == 1:
        return "T"
    elif 1 == c2:
        return "1P"
    elif 2 == c2:
        return "2P"
    else:
        return "TD"


def getMayor(array):
    valorMaximo = 0
    for element in array:
        if(element > valorMaximo):
            valorMaximo = element

    return valorMaximo


def normalizar(array):
    arrayNormal = np.array([])
    valorMaximo = getMayor(array)
    for element in array:
        arrayNormal = np.append(arrayNormal, element/valorMaximo)

    return arrayNormal


data = pd.read_excel("todo1.xlsx")
dataArray = np.array(data)
dataNormal = normalizar(dataArray)
n = len(dataArray)
manosDictio = {
    "TD": 0,
    "1P": 0,
    "2P": 0,
    "T": 0,
    "TP": 0,
    "P": 0,
    "Q": 0,
}
Ei = {
    "TD": 0.3024,
    "1P": 0.504,
    "2P": 0.108,
    "TP": 0.009,
    "T": 0.072,
    "P": 0.0045,
    "Q": 0.0001,
}
X = {
    "TD": 0,
    "1P": 0,
    "2P": 0,
    "T": 0,
    "TP": 0,
    "P": 0,
    "Q": 0,
}


for element in dataNormal:
    manosDictio[getMano(element)] += 1

for element in manosDictio:
    Ei[element] *= n

suma = 0
for i in X:
    X[i] = math.pow((Ei[i] - manosDictio[i]), 2) / Ei[i]
    suma += X[i]

alpha = 12.59 #Valor sacado de tablas

if suma < alpha:
    print(f"{suma} < {alpha}")
    print("No se rechaza la independencia de numeros")
if suma > alpha:
    print(f"{suma} > {alpha}")
    print("Se rechaza la independencia de numeros")
