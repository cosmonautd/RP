import sys
import time
import numpy
import argparse
from sklearn.model_selection import LeaveOneOut

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("--normalize", help="Normalizar os dados (zscore)", action="store_true")
ap.add_argument("-q", type=int, default=10, help="Número de neurônios da ELM")
args = ap.parse_args()

# Gerador aleatório com semente fixa para auxiliar na reproducibilidade
numpy.random.seed(1)

# Leitura da base de dados Dermatology
samples = list()
with open('dermatology.data') as derm:
    for row in derm.readlines():
        try: samples.append(list(map(float, row.split(','))))
        except ValueError: pass
dataset = numpy.array(samples)
# Preenchimento do vetor de amostras X e suas classes Y
X = dataset[:,0:len(dataset[0])-1]
# Representação de classes (e.g. [0 1 2])
Y_ = dataset[:,len(dataset[0])-1:].astype(int).flatten() - 1
# Representação one-hot-encoding (e.g. [[1 0 0], [0 1 0], [0 0 1]])
Y = numpy.zeros((X.shape[0], 6))
for i in range(len(Y_)): Y[i,Y_[i]] = 1

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

# Função sigmoide
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

# Normalização das amostras apenas se for passado parâmetro -n
if args.normalize:
    X = zscore(X)

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

# Instanciação do objeto responsável pela divisão de conjuntos de
# treino e teste de acordo com a metodologia Leave One Out
cross_val = LeaveOneOut()
cross_val.get_n_splits(X)

# Total de amostras
total = len(X)
# Matriz de confusão
conf_matrix = numpy.zeros((6, 6))

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X,Y_):

    # Assinala os conjuntos de treino e teste de acordo
    # com os índices definidos
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y_[test_index]

    # Realiza a inferência
    y = elm(X_train, Y_train, X_test, q=args.q)

    for i in range(len(y)):
        conf_matrix[y[i], Y_test[i]] += 1

# Cálculo do número de sucessos usando a matriz de confusão
# Soma dos elementos da diagonal principal
success = numpy.sum(numpy.diag(conf_matrix))

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('%.2f %%' % (result))

# Cálculo e empressão dos tempos médios de processamento
print('Tempo médio de treinamento: %f us' % (10**6*numpy.mean(t_train)))
print('Tempo médio de classificação: %f us' % (10**6*numpy.mean(t_classification)))

print('Matriz de confusão:')
print(conf_matrix)