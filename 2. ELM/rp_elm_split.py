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
    Z = tanh(numpy.dot(W, X_train))
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
    Z_ = tanh(numpy.dot(W, X_test))
    Z_ = numpy.concatenate((numpy.ones((1,N_)), Z_))
    # Cálculos dos valores da camada de saída
    D_ = tanh(numpy.dot(M, Z_))
    # Resultado da classificação
    for d in D_[0]:
        if d > 0: y.append(1)
        else: y.append(-1)
    y = numpy.array(y)
    # Fim da contagem de tempo e armazenamento do tempo transcorrido
    t_classification.append(time.time() - checkpoint)
    # Cálculo da superfície de separação
    # Criação de uma grade sobre a superfície dos dados
    x_0 = numpy.arange(-1, 8.5, 0.01)
    x_1 = numpy.arange(0, 7, 0.01)
    # Iteração sobre os pontos da grade
    boundary = list()
    for x0 in x_0:
        for x1 in x_1:
            # Classificação dos pontos da grade
            x = numpy.array([[1],[x0],[x1]])
            z = tanh(numpy.dot(W,x))
            z = numpy.concatenate((numpy.ones((1,1)), z))
            d = tanh(numpy.dot(M,z))
            # Caso a classificação seja incerta, registra os pontos
            # da grade que produziram a incerteza
            if abs(d) < 0.01:
                boundary.append([x0, x1])
    boundary = numpy.array(boundary)
    return y, boundary

# Realiza a inferência
y, b = elm(X, Y, X, q=args.q)

# Total de amostras
total = len(X)

# Realiza a contagem de sucessos
success = sum(y == Y.T[0])

# Cálculo e impressão do resultado da validação
result = 100*(success/total)
print('%.2f %%' % (result))

# Cálculo e empressão dos tempos médios de processamento
print('Tempo médio de treinamento: %f us' % (10**6*numpy.mean(t_train)))
print('Tempo médio de classificação: %f us' % (10**6*numpy.mean(t_classification)))

# Scatter plot dos dados de Two Moons
plt.scatter(X[:,0], X[:,1], c=Y.T[0], alpha=0.5)
# Plot da superfície de separação
plt.plot(b[:,0], b[:,1], 'red')
plt.show()