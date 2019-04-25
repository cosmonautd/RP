import sys
import time
import numpy
import argparse
import sklearn.cluster
from sklearn.model_selection import LeaveOneOut

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("--normalize", help="Normalizar os dados (zscore)", action="store_true")
ap.add_argument("-q", type=int, default=10, help="Número de neurônios da RBF")
args = ap.parse_args()

# Gerador aleatório com semente fixa para auxiliar na reproducibilidade
numpy.random.seed(1)

# Leitura da base de dados Iris
samples = list()
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)
# Preenchimento do vetor de amostras X e suas classes Y
X = dataset[:,0:len(dataset[0])-3]
# Representação one-hot-encoding (e.g. [[1 0 0], [0 1 0], [0 0 1]])
Y = dataset[:,len(dataset[0])-3:]
# Representação de classes (e.g. [0 1 2])
Y_ = numpy.argmax(Y, axis=1)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

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
        y.append(numpy.argmax(d, axis=0))
        # Fim da contagem de tempo de teste e armazenamento do tempo transcorrido
        t_classification.append(time.time() - checkpoint)
    y = numpy.array(y)
    return y

# Instanciação do objeto respY = numpy.argmax(Y, axis=1)onsável pela divisão de conjuntos de
# treino e teste de acordo com a metodologia Leave One Out
cross_val = LeaveOneOut()
cross_val.get_n_splits(X)

# Total de amostras
total = len(X)
# Variável para contagem da taxa de sucesso
success = 0.0

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X,Y_):

    # Assinala os conjuntos de treino e teste de acordo
    # com os índices definidos
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y_[test_index]

    # Realiza a inferência
    y = rbf(X_train, Y_train, X_test, q=args.q)

    # Realiza a contagem de sucessos
    success += sum(y == Y_test)

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('%.2f %%' % (result))

# Cálculo e empressão dos tempos médios de processamento
print('Tempo médio de treinamento: %f ms' % (10**3*numpy.mean(t_train)))
print('Tempo médio de classificação: %f us' % (10**6*numpy.mean(t_classification)))