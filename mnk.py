#Least square method
import math
import matplotlib.pyplot as plt
import numpy as np
import random as rand


def f1(x):
    return x

def f2(x):
    return x*x

def g(x,a,f):
    s=0
    for j in range(n):
        s=s+a[j]*f[j](x)
    return s

def noise(sigma, bias):
    r=sigma*(rand.random()-bias)
    return r



#Задаємо крок та межі відрізку
step = 0.05 
bound = 1.

x = np.arange(0., bound+step, step) #Обчислюємо масив значень X

#Моделювання
a=[2., 5.]
f=[f1,f2]

m=len(x)
n=len(f)

y=np.zeros(m)
ynar=np.zeros(m)

sigma=0.1
bias=0.5

for i in range(m):
    xx=x[i]
    y[i]=g(xx,a,f)
    ynar[i]=y[i]+noise(sigma,bias)
    

#Будуємо графіки
fig = plt.figure()
plt.plot(x,y,color='black')
plt.scatter(x,ynar,color='red')
plt.title('Теоретична функція та експериментальні дані')
plt.ylabel('Вісь Y')
plt.xlabel('Вісь X')
 
plt.grid(True)
 
 
plt.show()

#Отримання коефіцієнтів


v=np.zeros(m)

print("Length - ",m)



matr = [ [0]*n for i in range(m) ]

for i in range(m):
    xx=x[i]
    for j in range(n):
        matr[i][j]=f[j](xx)
    v[i]=ynar[i]

pv=np.linalg.pinv(matr);
acoefs=np.dot(pv,v)
print("Resulting vector:")
print(acoefs)


