import sys
import time
import keras
import numpy
import argparse
from sklearn.model_selection import LeaveOneOut

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("--normalize", help="Normalizar os dados (zscore)", action="store_true")
ap.add_argument("-q", type=int, default=10, help="Número de neurônios da MLP")
ap.add_argument("-e", type=int, default=20, help="Número de épocas de treinamento")
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

# Função MLP para classificar as amostras em X_test usando 
# X_train e Y_train como amostras de treino
def mlp(X_train, Y_train, X_test, q=10, epochs=1):
    # Lista de classes de saída
    y = list()
    # Cálculo do número p de atributos e número N de amostras
    p = X_train.shape[1]
    N = X_train.shape[0]
    # Início da contagem de tempo de treinamento
    checkpoint = time.time()
    # Instanciação de um modelo sequencial
    # Este modelo é uma pilha de camadas de neurônios
    # Sua construção é feita através da adição sequencial de camadas
    # A classe Dense representa camadas totalmente conectadas
    model = keras.models.Sequential([
        keras.layers.Dense(q, activation='sigmoid', input_shape=(p,)),
        keras.layers.Dense(6, activation='softmax')
    ])
    # Compilação do modelo
    # Definição do algoritmo de otimização e da função de perda
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # Treinamento
    # Executa o algoritmo de otimização, ajustando os pesos das conexões
    # da rede neural com base nos valores de entrada X_train e saída Y_train, 
    # usando a função de perda como forma de verificar o quão corretas são 
    # suas predições durante o treinamento.
    model.fit(X_train, Y_train, epochs=epochs, verbose=0)
    # Fim da contagem de tempo de treinamento e armazenamento 
    # do tempo transcorrido
    t_train.append(time.time() - checkpoint)
    # Início da contagem de tempo
    checkpoint = time.time()
    y_ = model.predict(X_test)
    y = numpy.argmax(y_, axis=1)
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
    y = mlp(X_train, Y_train, X_test, q=args.q, epochs=args.e)

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
print('Tempo médio de treinamento: %f ms' % (10**3*numpy.mean(t_train)))
print('Tempo médio de classificação: %f us' % (10**6*numpy.mean(t_classification)))

# Impressão da matriz de confusão
print('Matriz de confusão:')
print(conf_matrix)