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

def qda(X_train, Y_train, X_test):
    y = list()
    mu = list()
    cov_det = list()
    cov_inv = list()
    checkpoint = time.time()
    for class_ in sorted(list(set(Y_train))):
        idx = numpy.where(Y_train == class_)[0]
        mu.append(numpy.mean(X_train[idx], axis=0))
        cov = numpy.cov(X_train[idx], rowvar=False)
        cov_det.append(numpy.linalg.det(cov))
        cov_inv.append(numpy.linalg.inv(cov))
    t_train.append(time.time() - checkpoint)
    for x in X_test:
        checkpoint = time.time()
        p = list()
        for j in range(len(mu)):
            # p.append(numpy.log(numpy.linalg.det(cov[j])) + numpy.dot(numpy.dot((x - mu[j]).T, numpy.linalg.inv(cov[j])), x - mu[j]) - 2*numpy.log(1/3))
            p.append( (1/numpy.sqrt(2*numpy.pi*cov_det[j])) * numpy.exp( -0.5*numpy.dot(numpy.dot((x - mu[j]).T, cov_inv[j]), x - mu[j]) ) * (1/3))
        # y_ = numpy.argmin(p)
        y_ = numpy.argmax(p)
        t_classification.append(time.time() - checkpoint)
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
    y = one_nn(X_train, Y_train, X_test)

    # Realiza a contagem de sucessos
    success += sum(y == Y_test)

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('%.2f %%' % (result))

# Cálculo e empressão dos tempos médios de processamento
print('Tempo médio de treinamento: %f ms' % (1000*numpy.mean(t_train)))
print('Tempo médio de classificação: %f ms' % (1000*numpy.mean(t_classification)))