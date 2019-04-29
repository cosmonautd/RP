import numpy
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, default=1, help="Grau do polinômio de regressão")
args = ap.parse_args()

# Definição das coordenadas (x, y) na matrix X
X_ = numpy.array(
    [
        [122, 139],
        [114, 126],
        [ 86,  90],
        [134, 144],
        [146, 163],
        [107, 136],
        [ 68,  61],
        [117,  62],
        [ 71,  41],
        [ 98, 120]
    ]
)

# Definição das coordenadas z na matriz Z
Z_ = numpy.array(
    [
        [0.115],
        [0.120],
        [0.105],
        [0.090],
        [0.100],
        [0.120],
        [0.105],
        [0.080],
        [0.100],
        [0.115]
    ]
)

# Definição do grau do polinômio
k = args.k

# Produção da matriz X contendo as potências dos valores originais
# até o grau do polinômio de regressão definido; Z não é alterado
X = numpy.ones((len(X_), 1))
Z = Z_
for i in range(1, k+1):
    X = numpy.concatenate((X, numpy.power(X_, i)), axis=1)

# Cálculo dos coeficientes do plano regressor
beta = numpy.dot(numpy.linalg.pinv(numpy.dot(X.T, X)), numpy.dot(X.T, Z))

# Variáveis para cálculo do coeficiente de determinação R2
z = numpy.dot(X, beta)
Z_mean = numpy.mean(Z)
SQe = numpy.sum((Z - z)**2)
Syy = numpy.sum((Z - Z_mean)**2)

# Coeficiente de determinação R2
R2 = 1 - SQe/Syy

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot dos dados originais
ax.scatter(X[:,1].flatten(), X[:,2].flatten(), Z.flatten())

# Plot das previsões do modelo regressor sobre os dados originais
ax.scatter(X[:,1].flatten(), X[:,2].flatten(), numpy.dot(X, beta).flatten(), c='r')

# Configuração para plot do plano regressor, resolução e grade (x,y)
res = 20
x = numpy.arange(50, 180, (180 - 50)/res)
y = numpy.arange(50, 180, (180 - 50)/res)
x_grid, y_grid = numpy.meshgrid(x, y)

# Modelagem da superfície (x,y) sobre a qual serão calculados os valores de z
x_grid_flat = x_grid.reshape((res**2,1))
y_grid_flat = y_grid.reshape((res**2,1))
surface = numpy.ones((res**2,1))
for i in range(1, k+1):
    surface = numpy.concatenate(
        (surface, numpy.power(x_grid_flat, i), numpy.power(y_grid_flat, i)), axis=1)

# Aplicação do modelo regressor sobre a superfície de plot
z_grid = numpy.dot(surface, beta).reshape(x_grid.shape)

# Plot do plano regressor
ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.8)

# Configurações finais do plot
plt.title('Regressão 3D\nR2 = %.6f' % (R2))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()