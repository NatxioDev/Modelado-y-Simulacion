import numpy as np
import math
import random


def obtenerNumero(data, rand, n):
    acum = 0
    for i in range(1, n+1):
        if(rand >= acum and rand < data+acum):
            return i
        acum += data


def getLado():
    return obtenerNumero(1/6, random.random(), 6)


aux = np.array([])
dado = np.array([aux, aux, aux, aux, aux])
dado1 = np.array([])
dado2 = np.array([])
dado3 = np.array([])
dado4 = np.array([])
dado5 = np.array([])
for i in range(5):

    dado1 = np.append(dado1, getLado())
    dado2 = np.append(dado2, getLado())
    dado3 = np.append(dado3, getLado())
    dado4 = np.append(dado4, getLado())
    dado5 = np.append(dado5, getLado())

print("Dado 1: ", dado1)
print("Dado 2: ", dado2)
print("Dado 3: ", dado3)
print("Dado 4: ", dado4)
print("Dado 5: ", dado5)
