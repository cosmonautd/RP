import sys
import time
import numpy
import argparse
from sklearn.model_selection import StratifiedKFold

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("--normalize", help="Normalizar os dados (zscore)", action="store_true")
ap.add_argument("-e", type=int, default=20, help="Número de épocas de treinamento")
ap.add_argument("-a", type=float, default=0.25, help="Taxa de aprendizagem")
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

# Normalização das amostras apenas se for passado parâmetro -n
if args.normalize:
    X = zscore(X)

# Listas para armazenar tempos de treinamento e classificação
t_train = list()
t_classification = list()

# Função Perceptron para classificar as amostras em X_test usando
# X_train e Y_train como amostras de treino
def perceptron(X_train, Y_train, X_test, epochs, alpha):
    # Lista de classes de saída
    y = list()
    # Configurações iniciais do número de amostras N, número 
    # de atributos p, número de neurônios q e lista de pesos 
    # para cada neurônio
    N = X_train.shape[0]
    p = X_train.shape[1]
    q = 6
    W = list()
    # Início da contagem de tempo de treinamento
    checkpoint = time.time()
    # Adição do bias à matriz X_train
    X_train = numpy.concatenate((-numpy.ones((N,1)), X_train), axis=1)
    # Treinamento de cada neurônio da rede perceptron individualmente
    for q_ in range(q):
        # Inicialização dos pesos do neurônio em questão
        W.append(numpy.zeros((1, p+1)))
        # Passagem de épocas
        for epoch in range(epochs):
            # Iteração pelas amostras de treinamento
            for i, x in enumerate(X_train):
                # Cálculo da ativação
                u_t = numpy.dot(W[q_].flatten(), x)
                # Cálculo da saída do neurônio
                y_t = 1 if u_t > 0 else 0
                # Cálculo do erro
                # Caso 1: o neurônio não ativa, mas deveria ativar
                if y_t == 0 and Y_train[i] == q_: e_t = 1
                # Caso 2: o neurônio ativa, mas não deveria ativar
                elif y_t == 1 and Y_train[i] != q_: e_t = -1
                # Caso 3: o neurônio ativa ou não ativa corretamente
                elif y_t == 0 and Y_train[i] != q_: e_t = 0
                elif y_t == 1 and Y_train[i] == q_: e_t = 0
                # Atualização dos pesos do neurônio em questão
                W[q_] = W[q_] + alpha*e_t*x
    # Fim da contagem de tempo de treinamento e armazenamento 
    # do tempo transcorrido
    t_train.append(time.time() - checkpoint)
    # Obtenção do número de amostras a serem classificadas
    N_ = X_test.shape[0]
    # Início da contagem de tempo
    checkpoint = time.time()
    # Adição do bias à matriz X_test
    X_test = numpy.concatenate((-numpy.ones((N_,1)), X_test), axis=1)
    # Iteração por todas as amostras de teste
    for i, x in enumerate(X_test):
        # Saída da rede perceptron para a amostra de teste atual
        y_ = list()
        # Iteração pelos neurônios da rede
        for q_ in range(q):
            # Cálculo da ativação no neurônio
            u_t = numpy.dot(W[q_].flatten(), x)
            # Cálculo da saída do neurônio
            y_q = 1 if u_t > 0 else 0
            # Armazenamento da saída do neurônio no array de saída da rede
            y_.append(y_q)
        y_ = numpy.array(y_)
        # Caso apenas um neurônio tenha sido ativado, armazena o índice
        # do neurônio como a classe da amostra; caso contrário, o índice
        # -1 é usado como indicação de que a rede não pôde classificar a amostra
        if numpy.sum(y_) == 1: y.append(numpy.argmax(y_))
        else: y.append(-1)
    y = numpy.array(y)
    # Fim da contagem de tempo e armazenamento do tempo transcorrido
    t_classification.append(time.time() - checkpoint)
    return y

# Instanciação do objeto responsável pela divisão de conjuntos de
# treino e teste de acordo com a metodologia K-Fold com K = 5
cross_val = StratifiedKFold(5)
cross_val.get_n_splits(X)

# Total de amostras
total = len(X)
# Matriz de confusão
conf_matrix = numpy.zeros((7, 7))

# Percorre as divisões de conjuntos de treino e teste 
# 5-Fold
for train_index, test_index in cross_val.split(X,Y_):

    # Assinala os conjuntos de treino e teste de acordo
    # com os índices definidos
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y_[train_index], Y_[test_index]

    # Realiza a inferência
    y = perceptron(X_train, Y_train, X_test, epochs=args.e, alpha=args.a)

    # Preenche a matriz de confusão
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

# Impressão da matriz de confusão
print('Matriz de confusão:')
print(conf_matrix)