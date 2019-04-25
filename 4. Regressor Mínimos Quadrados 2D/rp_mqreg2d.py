import numpy
import argparse
import matplotlib.pyplot as plt

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, default=4, help="Grau do polinômio de regressão")
args = ap.parse_args()

# Leitura da base de dados do Aerogerador
samples = list()
with open('aerogerador.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)

# Preenchimento do vetor de amostras original X_ e suas classes Y_
X_ = dataset[:,0].reshape((len(dataset), 1))
Y_ = dataset[:,1].reshape((len(dataset), 1))

# Número de amostras do dataset
n = len(dataset)

# Definição do grau do polinômio
k = args.k

# Produção da matriz X contendo as potências dos valores originais
# até o grau do polinômio de regressão definido; Y não é alterado
X = numpy.ones((len(X_), 1))
Y = Y_
for i in range(1, k+1):
    X = numpy.concatenate((X, numpy.power(X_, i)), axis=1)

# Cálculo dos coeficientes do polinômio de regressão
beta = numpy.dot(numpy.linalg.pinv(numpy.dot(X.T, X)), numpy.dot(X.T, Y))

# Variáveis para cálculo dos coeficientes de determinação R2 e R2aj
y = numpy.dot(X, beta)
Y_mean = numpy.mean(Y)
p = k + 1
SQe = numpy.sum((Y - y)**2)
Syy = numpy.sum((Y - Y_mean)**2)

# Coeficientes de determinação R2 e R2aj
R2 = 1 - SQe/Syy
R2aj = 1 - ((SQe/(n-p))/(Syy/(n-1)))

# Resolução e cálculo do intervalo do plot
res = 1000
min_x = min(X_)
max_x = max(X_)
x_plot = numpy.arange(min_x, max_x, (max_x - min_x)/res).reshape((res, 1))

# Aplicação do polinômio de regressão para o intervalo do plot
x  = numpy.ones((1000, 1))
for i in range(1, k+1):
    x = numpy.concatenate((x, numpy.power(x_plot, i)), axis=1)
y_plot = numpy.dot(x, beta)

# Plot dos dados e da curva de regressão
plt.title('Regressão MQ sobre os dados do aerogerador com k = %d' % (k))
plt.xlabel('Velocidade do vento (m/s)')
plt.ylabel('Potência gerada (kW)')
plt.scatter(X_, Y_, alpha=0.5, label='Dados do aerogerador')
plt.plot(x_plot, y_plot, color='red', label='Curva de regressão')
plt.annotate('R2    = %.6f' % (R2), xy=(0.025, 0.80), xycoords='axes fraction')
plt.annotate('R2aj = %.6f' % (R2aj), xy=(0.025, 0.75), xycoords='axes fraction')
plt.legend()
plt.savefig('reg%d.pdf' % (k))
plt.show()