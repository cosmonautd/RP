import time
import numpy
from sklearn.model_selection import LeaveOneOut

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

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

# Normalização das amostras
X = zscore(X)

# Listas para armazenar tempos de treinamento e classificação
t_train = list()
t_classification = list()

# Função QDA para treinar e classificar as amostras em X_test
# usando X_train e Y_train como amostras conhecidas
def qda(X_train, Y_train, X_test):
    # Lista de classes de saída
    y = list()
    # Lista de centroides, um para cada classe
    mu = list()
    # Lista de determinates das matrizes de covariâncias, um para cada classe
    cov_det = list()
    # Lista de inversas das matrizes de covariâncias, uma para cada classe
    cov_inv = list()
    # Início da contagem de tempo de treinamento
    checkpoint = time.time()
    # Percorre o conjunto de classes na base de dados
    for class_ in sorted(list(set(Y_train))):
        # Separa os índices do elementos cuja classe atual
        idx = numpy.where(Y_train == class_)[0]
        # Calcula e armazena o centroide da classe atual
        mu.append(numpy.mean(X_train[idx], axis=0))
        # Calcula a matriz de covariâncias da classe atual
        cov = numpy.cov(X_train[idx], rowvar=False)
        # Calcula e armazena o determinante da matriz de covariâncias da classe atual
        cov_det.append(numpy.linalg.det(cov))
        # Calcula e armazena a inversa da matriz de covariâncias da classe atual
        cov_inv.append(numpy.linalg.inv(cov))
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
            p.append(numpy.log(cov_det[j]) +
                     numpy.dot(numpy.dot((x - mu[j]).T, cov_inv[j]), x - mu[j]))
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
    y = qda(X_train, Y_train, X_test)

    # Realiza a contagem de sucessos
    success += sum(y == Y_test)

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('%.2f %%' % (result))

# Cálculo e empressão dos tempos médios de processamento
print('Tempo médio de treinamento: %f ms' % (1000*numpy.mean(t_train)))
print('Tempo médio de classificação: %f ms' % (1000*numpy.mean(t_classification)))