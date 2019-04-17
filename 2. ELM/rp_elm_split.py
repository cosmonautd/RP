import time
import numpy
import argparse
import matplotlib.pyplot as plt

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("--normalize", help="Normalizar os dados (zscore)", action="store_true")
ap.add_argument("-q", type=int, default=10, help="Número de neurônios da ELM")
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
# Representação de classes (e.g. [0 1])
Y_ = dataset[:,2:]
Y_ = [0 if y == -1 else 1 for y in Y_]
Y_ = numpy.array(Y_)
# Representação one-hot-encoding (e.g. [[1 0], [0 1]])
Y = [[1,0] if y == 0 else [0,1] for y in Y_]
Y = numpy.array(Y)

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
    D_ = sigmoid(numpy.dot(M, Z_))
    # Cálculo do índice do neurônio que proporciona a maior ativação na saída
    y = numpy.argmax(D_, axis=0)
    # Fim da contagem de tempo e armazenamento do tempo transcorrido
    t_classification.append(time.time() - checkpoint)
    return y, s

# Realiza a inferência
y, s = elm(X, Y, X, q=args.q)

# Total de amostras
total = len(X)

# Realiza a contagem de sucessos
success = sum(y == Y_)

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('%.2f %%' % (result))

# Cálculo e empressão dos tempos médios de processamento
print('Tempo médio de treinamento: %f us' % (10**6*numpy.mean(t_train)))
print('Tempo médio de classificação: %f us' % (10**6*numpy.mean(t_classification)))

plt.scatter(X[:,0], X[:,1], c=Y_, alpha=0.5)
plt.plot(s[:,0], s[:,1])
plt.show()