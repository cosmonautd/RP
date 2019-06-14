import time
import numpy
import argparse
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

# Ajuste de formatação dos valores em arrays numpy
numpy.set_printoptions(formatter={'float': lambda x: '%5.2f' % x})

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

# Função para transformação PCA com utilização
# de c componentes principais
def pca(X, c=None):
    # Caso o valor de c não seja passado, utilizar valor
    # igual ao número original de dimensões
    if c is None: c = X.shape[1]
    # Caso o número de componentes solicitado seja 0, não realiza PCA
    if c == 0: return X
    # Subtração da média dos atributos
    X = X - numpy.mean(X, axis=0)
    # Cálculo da matriz de covariâncias
    cov = numpy.cov(X, rowvar=False)
    # Cálculo dos autovalores e autovetores
    eigenvalues, eigenvectors = numpy.linalg.eig(cov)
    # Ordenação decrescente dos autovetores de acordo
    # com os autovalores
    eig = list(zip(eigenvalues, eigenvectors.T))
    eig.sort(key=lambda eig: eig[0], reverse=True)
    # Reconstrução da matriz composta de autovetores ordenados
    A = numpy.array([ev[1] for ev in eig[:c]]).T
    # Multiplicação da base original pela matriz de autovetores
    return numpy.dot(X, A)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

# Função sigmoide
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

# Listas para armazenar tempos de treinamento e classificação
t_train = list()
t_classification = list()

# Função ELM para classificar as amostras em X_test usando 
# X_train e Y_train como amostras de treino
def elm(X_train, Y_train, X_test, q=10):
    # Lista de classes de saída
    y = list()
    # Alteração do padrão dos dados (amostras são posicionadas nas colunas)
    X_test = X_test.T
    X_train = X_train.T
    D = Y_train.T
    # Cálculo do número p de atributos e número N de amostras
    p = X_train.shape[0]
    N = X_train.shape[1]
    # Início da contagem de tempo de treinamento
    checkpoint = time.time()
    # Geração da matriz de pesos aleatórios
    W = numpy.random.randn(q, p+1)
    # Adição do bias à matriz X_train
    X_train = numpy.concatenate((numpy.ones((1,N)), X_train))
    # Passagem pela camada oculta e adição de bias a Z
    Z = sigmoid(numpy.dot(W, X_train))
    Z = numpy.concatenate((numpy.ones((1,N)), Z))
    # Cálculo dos pesos entre a camada oculta e camada de saída
    M = numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.pinv(numpy.dot(Z,Z.T))))
    # Fim da contagem de tempo de treinamento e armazenamento 
    # do tempo transcorrido
    t_train.append(time.time() - checkpoint)
    # Obtenção do número de amostras a serem classificadas
    N_ = X_test.shape[1]
    # Início da contagem de tempo
    checkpoint = time.time()
    # Adição do bias
    X_test = numpy.concatenate((numpy.ones((1,N_)), X_test))
    # Passagem das amostras de teste pela camada oculta e adição do bias
    Z_ = sigmoid(numpy.dot(W, X_test))
    Z_ = numpy.concatenate((numpy.ones((1,N_)), Z_))
    # Cálculos dos valores da camada de saída
    D_ = numpy.dot(M, Z_)
    # Cálculo do índice do neurônio que proporciona a maior ativação na saída
    y = numpy.argmax(D_, axis=0)
    # Fim da contagem de tempo e armazenamento do tempo transcorrido
    t_classification.append(time.time() - checkpoint)
    return y

# Matriz para armazentamento dos resultados
nlist = [0, 1, 2, 3, 4]
Qlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
R = numpy.zeros((len(nlist), len(Qlist)))

# Iteração pelos valores desejados de número de componentes principais
for n in nlist:
    # Iteração pelos valores desejados de número de neurônios na camada oculta
    for Q in Qlist:
        # Aplicação do PCA
        X_pca = pca(X, n)
        # Instanciação do objeto responsável pela divisão de conjuntos de
        # treino e teste de acordo com a metodologia Leave One Out
        cross_val = LeaveOneOut()
        cross_val.get_n_splits(X_pca)
        # Total de amostras
        total = len(X_pca)
        # Variável para contagem da taxa de sucesso
        success = 0.0
        # Percorre as divisões de conjuntos de treino e teste
        # Leave One Out
        for train_index, test_index in cross_val.split(X_pca,Y):
            # Assinala os conjuntos de treino e teste de acordo
            # com os índices definidos
            X_train, X_test = X_pca[train_index], X_pca[test_index]
            Y_train, Y_test = Y[train_index], Y_[test_index]
            # Realiza a inferência
            y = elm(X_train, Y_train, X_test, q=Q)
            # Realiza a contagem de sucessos
            success += sum(y == Y_test)
        # Cálculo e impressão do resultado da validação
        result = 100*(success/total)
        R[n, Q-1] = result

ax = seaborn.heatmap(R, cmap='YlGnBu', annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.41})
ax.set_xlabel('Neurônios na camada oculta (Q)')
ax.set_ylabel('Componentes principais (n)')
ax.set_xticklabels(['%d' % (Q) for Q in Qlist])
ax.set_yticklabels(['%d' % (n) for n in nlist], rotation=0)
plt.tight_layout()
plt.show()