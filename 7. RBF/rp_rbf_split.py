import time
import numpy
import argparse
import sklearn.cluster
import matplotlib.pyplot as plt

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("--normalize", help="Normalizar os dados (zscore)", action="store_true")
ap.add_argument("-q", type=int, default=10, help="Número de neurônios da RBF")
args = ap.parse_args()

# Gerador aleatório com semente fixa para auxiliar na reproducibilidade
numpy.random.seed(1)

# Leitura da base de dados Two Moons
samples = list()
with open('twomoons.dat') as twomoons:
    for row in twomoons.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)
# Preenchimento do vetor de amostras X e suas classes Y
X = dataset[:,0:2]
# Representação de classes (e.g. [1 -1])
Y = dataset[:,2:]

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

def tanh(x):
    return numpy.tanh(x)

# Normalização das amostras apenas se for passado parâmetro -n
if args.normalize:
    X = zscore(X)

# Listas para armazenar tempos de treinamento e classificação
t_train = list()
t_classification = list()

# Função RBF para classificar as amostras em X_test usando 
# X_train e Y_train como amostras de treino
def rbf(X_train, Y_train, X_test, q=10):
    # Lista de classes de saída, matriz de rótulos e número de amostras
    y = list()
    D = Y_train.T
    n = len(X_train)
    # Início da contagem de tempo de treinamento
    checkpoint = time.time()
    # Cálculo dos centroides usando K-Means com q clusters
    kmeans = sklearn.cluster.KMeans(n_clusters=q).fit(X_train)
    t = kmeans.cluster_centers_
    # Cálculo das ativações da camada oculta da RBF
    Z = list()
    for x in X_train:
        # Uso da função de base radial
        Z.append(numpy.exp((-numpy.linalg.norm(x-t, axis=1)**2)))
    # Configuração das dimensões da matriz Z e adição do bias
    Z = numpy.array(Z).T
    Z = numpy.concatenate((numpy.ones((1,n)), Z), axis=0)
    # Cálculo dos pesos da camada de saída usando Mínimos Quadrados
    M = numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.pinv(numpy.dot(Z,Z.T))))
    # Fim da contagem de tempo de treinamento e armazenamento do tempo transcorrido
    t_train.append(time.time() - checkpoint)
    for x in X_test:
        # Início da contagem de tempo de teste
        checkpoint = time.time()
        # Passagem da amostra x pela camada oculta
        z = numpy.exp(-numpy.linalg.norm(x-t, axis=1)**2)
        # Adição do bias
        z = numpy.concatenate((numpy.ones(1), z))
        # Passagem pelos pesos da camada de saída
        d = numpy.dot(M, z)
        # Cálculo do índice do neurônio que proporciona a maior ativação na saída
        y.append(-1 if d < 0 else 1)
        # Fim da contagem de tempo de teste e armazenamento do tempo transcorrido
        t_classification.append(time.time() - checkpoint)
    y = numpy.array(y)
    # Cálculo da superfície de separação
    # Criação de uma grade sobre a superfície dos dados
    x_0 = numpy.arange(-2, 8.5, 0.01)
    x_1 = numpy.arange(0, 8, 0.01)
    # Iteração sobre os pontos da grade
    boundary = list()
    for x0 in x_0:
        for x1 in x_1:
            # Classificação dos pontos da grade
            x = numpy.array([x0, x1])
            # Passagem da amostra x pela camada oculta
            z = numpy.exp(-numpy.linalg.norm(x-t, axis=1)**2)
            # Adição do bias
            z = numpy.concatenate((numpy.ones(1), z))
            # Passagem pelos pesos da camada de saída
            d = numpy.dot(M, z)
            # Caso a classificação seja incerta, registra os pontos
            # da grade que produziram a incerteza
            if abs(d) < 0.005:
                boundary.append([x0, x1])
    boundary = numpy.array(boundary)
    return y, boundary

# Realiza a inferência
y, b = rbf(X, Y, X, q=args.q)

# Total de amostras
total = len(X)

# Realiza a contagem de sucessos
success = sum(y == Y.T[0])

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('Acurácia: %.2f %%' % (result))

# Scatter plot dos dados de Two Moons
plt.scatter(X[:,0], X[:,1], c=Y.T[0], alpha=0.5)
# Plot da superfície de separação
plt.scatter(b[:,0], b[:,1], s=1.1, c='red', alpha=0.25)
plt.tight_layout()
plt.show()