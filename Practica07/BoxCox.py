import pandas as pd
import matplotlib.pyplot as plt
import math as m
import numpy as np

plt.close("all")
fileData = pd.read_csv('data.csv')
promedio = fileData.iloc[:, 0]
edad = fileData.iloc[:, 1]  # este
nmaterias = fileData.iloc[:, 2]  # este
hestudio = fileData.iloc[:, 3]  # este

xejercicio = []
y = []

x1 = edad.mean()
x2 = nmaterias.mean()
x3 = hestudio.mean()
sumx1 = edad.sum()
sumx2 = nmaterias.sum()
sumx3 = hestudio.sum()
sumx1cuadrado = (edad*edad).sum()
sumx2cuadrado = (nmaterias*nmaterias).sum()
sumx3cuadrado = (hestudio*hestudio).sum()
sumx2porx1 = (nmaterias*edad).sum()
sumx3porx1 = (hestudio*edad).sum()
sumx3porx2 = (hestudio*nmaterias).sum()
yprom = promedio.mean()
y1 = promedio.sum()
y2 = (promedio*edad).sum()
y3 = (promedio*nmaterias).sum()
y4 = (promedio*hestudio).sum()

xejercicio = [[1, sumx1, sumx2, sumx3],
              [sumx1, sumx1cuadrado, sumx2porx1, sumx3porx1],
              [sumx2, sumx2porx1, sumx2cuadrado, ],
              [sumx3, sumx3porx1, sumx3porx2, sumx3cuadrado]]
y = [[y1], [y2], [y3], [y4]]
beta3 = 1/(x1*sumx1*((x3*x3)*sumx3+1))
beta2 = ((beta3*x3*((sumx3*(1-x1)+x2*(1-sumx1))))/((sumx1-1)*(x2-sumx2)))
beta1 = ((beta2*x2*(x1-sumx1)+(beta3*x3*(x1-sumx1)))/((x1*sumx1)-sumx1cuadrado))
beta0 = yprom - beta1*x1-beta2*x2-beta3*x3

beta = [[beta0], [beta1], [beta2], [beta3]]
print("........................")
print("ECUACION")
print("y = ", beta0, " + ", beta1, " X1  +", beta2, " X2 + ", beta3, " X3")
print("........................")

ynorm = fileData.iloc[:, 0:1]
yarray = np.array(ynorm)
x = fileData.iloc[:, 1:4]
u = np.array(np.ones(37))
xarray = np.array(x)
xarray = np.insert(xarray, 0, 1, axis=1)
xt = np.transpose(xarray)
c = np.matmul(xt, xarray)
#print("Valor de c", c)
cinv = np.linalg.inv(c)
#print("Valor de cinv", cinv)
xtpory = np.matmul(xt, yarray)
beta_estimado = np.matmul(cinv, xtpory)
print("beta estimado", beta_estimado)
print("........................")
yestimado = np.matmul(xarray, beta_estimado)
k = len(beta_estimado)
n = len(yarray)
varianza = (((yarray-yestimado)**2).sum())/(n-k)

v = []

print("varianza: ", varianza)
print("........................")
desviacion = m.sqrt(varianza)

print("VARIANZA POR Cii")

for i in range(0, len(c), 1):
    for j in range(0, len(c), 1):
        if i == j:
            vporc = varianza*cinv[i][j]
            print("valor ", i, j, vporc)
            desv = desviacion*m.sqrt(cinv[i][j])
            print("valor desviacion estandar ", i, j, " : ", desv)
            v.append(desv)
print("........................")
#print (v)
rango = 4
deltalamda = 0.1
numeroval = rango/deltalamda
delta = np.array([-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -
                  0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
#print (delta)
n = numeroval+1
final = n**rango
print(final)
final = int(final)
print(final)
#final = 10
# espacio para el calculo del box cox
delta1 = 0
delta2 = 0
delta3 = 0
y1 = 0
ydatos = []
for i in range(0, final, 1):
    for j in range(0, len(delta), 1):
        if (delta[j] == 0):
            delta1 = np.log(xarray[1])
            delta2 = np.log(xarray[2])
            delta3 = np.log(xarray[3])
        else:
            delta1 = (((xarray[1]**delta[j])-1)/delta[j])
            delta2 = (((xarray[2]**delta[j])-1)/delta[j])
            delta3 = (((xarray[3]**delta[j])-1)/delta[j])

        ybox = beta_estimado[0]*xarray[0] + beta_estimado[1] * \
            delta1 + beta_estimado[2]*delta2 + beta_estimado[3]*delta3
        # print ("Recorrido ",j," = ", ybox)
        ydatos.append(ybox)

    print(i)
print(len(ydatos))


# espacio para hallar las t-student
v = np.array(v)
# print("beta",beta)
Ti = []
for i in range(0, len(beta), 1):

    t = ((beta_estimado[i]-beta[i])/(v[i]))
    Ti.append(t)
print("VALOR DE Ti: ", Ti)
print("........................")
print("GRADOS DE LIBERTAD ", 37-k)
# t 2.34834
t = float(input("Ingrese el valor de las tablas de t student "))
# t=3.29
print("........................")
t1 = (t*v.mean())+beta_estimado.mean()
t2 = beta_estimado.mean()-(t*v.mean())
print("EL LIC: ", t2)
print("EL LSC: ", t1)
print("........................")
for i in range(0, len(beta_estimado), 1):
    if beta_estimado[i] == 0:
        print("Se elimina ", beta_estimado[i])
    else:
        print("No se elimina ", beta_estimado[i])

print("........................")
for i in range(0, len(beta_estimado), 1):
    if (beta_estimado[i]/v[i]) > t:
        print("Se rechaza la hipotesis nula de: ", beta_estimado[i]/v[i])
        print("Se rechaza la hipotesis nula de: ", Ti[i])
    else:
        print("No se rechaza la hipotesis nula de: ", beta_estimado[i]/v[i])
        print("No Se rechaza la hipotesis nula de: ", Ti[i])

yestimadoprom = yestimado.mean()
yprom = yarray.mean()
sec = ((yestimado-yestimadoprom)*(yestimado-yestimadoprom)).sum()
src = ((yarray-yestimado)*(yarray-yestimado)).sum()
stc = ((yarray-yprom)*(yarray-yprom)).sum()
# k=len(beta_estimado);
print("........................")
print("Analisis de modelo con prueba F-fisher")
valor = beta_estimado[0]
cont = -1
for i in range(0, len(beta_estimado), 1):
    if valor == beta_estimado[i]:
        cont = cont+1

if (cont == len(beta_estimado)):
    print("Los valores betas son iguales, no son significantes para el modelo")
    print("el modelo se debe eliminar")
else:
    print("Los valores betas no on iguales, son significantes para el modelo")
    print("el modelo no se debe eliminar")

gld = n-k
gln = k
print("........................")

print("Grados de libertad denominador ", gld)
print("grados de libertad numerador", gln)
print("........................")
sc1 = sec/k
sc2 = src / (n-k)
print("valor en numerador de sc: ", sc1)
print("valor en denominador de sc: ", sc2)
print("........................")
F1 = sc1/sc2
print("valor de F obtenido con sc1 / sc2: ", F1)
#f: 0.37610
Ftablas = float(input("Ingrese el valor de las tablas de F-fisher  "))
if (F1 > Ftablas):
    print("se rechaza la hipotesis nula")
else:
    print("No se rechaza la hipotesis nula")
print("........................")
