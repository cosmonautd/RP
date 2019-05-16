import time
import numpy
import argparse
from sklearn.model_selection import LeaveOneOut

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-c", type=int, default=1, help="Número de componentes do PCA")
args = ap.parse_args()

# Leitura da base de dados Iris
samples = list()
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)
# Preenchimento do vetor de amostras X e suas classes Y
X = dataset[:,0:len(dataset[0])-3]
Y = dataset[:,len(dataset[0])-3:]
Y = numpy.argmax(Y, axis=1)

# Função para transformação PCA com utilização
# de c componentes principais
def pca(X, c=None):
    # Caso o valor de c não seja passado, utilizar valor
    # igual ao número original de dimensões
    if c is None: c = X.shape[1]
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

# Caso o número de componentes passado seja positivo, realiza o PCA
if args.c > 0:
    X = pca(X, args.c)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

X = zscore(X)

# Listas para armazenar tempos de treinamento e classificação
t_train = list()
t_classification = list()

# Função LDA para treinar e classificar as amostras em X_test
# usando X_train e Y_train como amostras conhecidas
def lda(X_train, Y_train, X_test):
    # Lista de classes de saída
    y = list()
    # Lista de centroides, um para cada classe
    mu = list()
    # Início da contagem de tempo de treinamento
    checkpoint = time.time()
    # Cálculo da matriz de covariâncias sobre todas as amostras de treinamento
    cov = numpy.cov(X_train, rowvar=False)
    # Cálculo da inversa da matriz de covariâncias
    try: cov_inv = numpy.linalg.inv(cov)
    except: cov_inv = 1/cov
    # Percorre o conjunto de classes na base de dados
    for class_ in sorted(list(set(Y_train))):
        # Separa os índices do elementos cuja classe atual
        idx = numpy.where(Y_train == class_)[0]
        # Calcula e armazena o centroide da classe atual
        mu.append(numpy.mean(X_train[idx], axis=0))
    # Fim da contagem de tempo de treinamento e armazenamento 
    # do tempo transcorrido
    t_train.append(time.time() - checkpoint)
    # Percorre o conjunto de elementos a serem classificados
    for x in X_test:
        # Início da contagem de tempo
        checkpoint = time.time()
        # Lista de valores discriminantes, uma para cada classe
        p = list()
        # Percorre as classes presentes no problema usando como referência
        # a lista de centroides já calculados
        for j in range(len(mu)):
            # Calcula e armazena o resultado do discriminante associado à classe j
            p.append(numpy.dot(numpy.dot((x - mu[j]).T, cov_inv), x - mu[j]))
        # Cálculo do índice da classe que minimiza o valor do discriminante
        y_ = numpy.argmin(p)
        # Fim da contagem de tempo e armazenamento do tempo transcorrido
        t_classification.append(time.time() - checkpoint)
        # O resultado da classificação é armazenado
        y.append(y_)
    return numpy.array(y)

# Instanciação do objeto responsável pela divisão de conjuntos de
# treino e teste de acordo com a metodologia Leave One Out
cross_val = LeaveOneOut()
cross_val.get_n_splits(X)

# Total de amostras
total = len(X)
# Variável para contagem da taxa de sucesso
success = 0.0

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X,Y):

    # Assinala os conjuntos de treino e teste de acordo
    # com os índices definidos
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Realiza a inferência
    y = lda(X_train, Y_train, X_test)

    # Realiza a contagem de sucessos
    success += sum(y == Y_test)

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('%.2f %%' % (result))

# Cálculo e empressão dos tempos médios de processamento
print('Tempo médio de treinamento: %f us' % (10**6*numpy.mean(t_train)))
print('Tempo médio de classificação: %f us' % (10**6*numpy.mean(t_classification)))