import numpy as np
import pandas as pd
import math
import copy


def getSCT(y, n):
    meany = y.sum()/n
    # print(meany)
    suma = 0
    for i in range(n):
        suma = suma+((y[i, 0]-meany)**2)
    return suma


def regMult(x, y, n, k, lam):
    xc = copy.copy(x)
    yc = copy.copy(y)
    for i in range(n):
        for j in range(k):
            if lam[j] == 0:
                return 0, lam
            xc[i, j+1] = ((xc[i, j+1]**lam[j])-1)/lam[j]
    xt = xc.T
    xtx = xt*xc
    xtxi = np.linalg.inv(xtx)
    xtxixt = xtxi*xt
    beta = xtxixt*y
    lu = (xc*xtxi*xc.T)
    sce = (yc.T)*(np.identity(n)-lu)*yc
    unos = np.ones(n, order='C')
    cunos = unos.reshape((n, 1))
    cunost = unos.reshape((1, n))
    yest = xc*beta
    lunos = cunos*(1/n)*cunos
    if intercepto == 0:
        scr = (yc.T)*(lu)*yc
    else:
        scr = (yc.T)*(lu-lunos)*yc
    if intercepto == 0:
        den = sce/(n-k)
    else:
        den = sce/(n-k-1)
    num = scr/(k)
    t = float(den)*xtxi
    if intercepto == 1:
        k = k+1
    for i in range(k):
        cof = math.sqrt(float(t[i, i]))
        tstud = float(beta[i, 0])/cof
    rcuad = float(scr/getSCT(yc, n))
    return rcuad, beta


def generateLam(i, linf):
    lam = np.ones(k)
    cont = 0
    while i > 0:
        lam[cont] = linf+((i % 21)*0.1)
        cont = cont+1
        i = int(i/21)
    return lam


n = 8  # Cantidad de Datos Observados
k = 3  # Cantidad de Variables Independientes
intercepto = 1
bd = pd.read_csv("datax1.csv")  # Nombre del archivo de entrada de x
bd2 = pd.read_csv("datay1.csv")  # Nombre del archivo de entrada de y
x = np.matrix(bd)
y = np.matrix(bd2)

# definimos rangos de lambda
linf = -2
lsup = 2
delta = 0.1
rcuadgan = -10
iterac = k**21
for i in range(k):
    lam = generateLam(i, linf)
    rcuadrado, cbeta = regMult(x, y, n, k, lam)
    if rcuadrado > rcuadgan:
        lamgan = lam
        betagan = cbeta
        rcuadgan = rcuadrado
print(rcuadgan)
print(lamgan)
print(betagan)



for i in range(k):
    lam = np.ones(k)
    cont = 0
    while i > 0:
        lam[cont] = linf+((i % 21)*0.1)
        cont = cont+1
        i = int(i/21)
    
    rcuadrado, cbeta = regMult(x, y, n, k, lam)
    if rcuadrado > rcuadgan:
        lamgan = lam
        betagan = cbeta
        rcuadgan = rcuadrado
