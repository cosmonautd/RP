import time
import numpy
import argparse
import random
import seaborn
import matplotlib.pyplot as plt

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, default=None, help="Número de clusters")
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
# Representação de classes (e.g. [0 1 2])
Y = numpy.argmax(Y, axis=1)

# Função de clustering K-means
def kmeans(X, k):
    # Obtenção do número de amostras N e sua dimensionalidade d
    N, d = X.shape
    # Os centroides iniciais são elementos do dataset escolhidos aleatoriamente
    centroids = numpy.array(random.sample(list(X), k))
    # Lista contendo o identificador do cluster associado a cada amostra
    clusters = numpy.zeros(N).astype(int)
    # Início do loop
    while True:
        # Para cada amostra do dataset
        for i, x in enumerate(X):
            # Calcula a distância entre a amostra e os centroides
            dist = numpy.linalg.norm(x - centroids, axis=1)
            # Determina o centroide cuja distância à amostra é mínima
            m = numpy.argmin(dist)
            # Assinala a amostra ao cluster cuja distância do centroide até a amostra é mínima
            clusters[i] = m
        # Cria uma cópia da lista de centroides
        centroids_ = numpy.copy(centroids)
        # Para cada centroide/cluster
        for j, _ in enumerate(centroids):
            # Seleciona as amostras assinaladas ao centroide/cluster
            z = numpy.array([x for i, x in enumerate(X) if clusters[i] == j])
            # Se existirem amostras no cluster sendo processado atualmente
            if len(z) > 0:
                # Atualiza o centroide, substituindo-o pela média das amostras em seu cluster
                centroids[j] = numpy.mean(z, axis=0)
        # Critério de parada: os centroide não sofreram atualização
        if numpy.all(centroids == centroids_): break
    # Retorna os clusters associados a cada amostra e a lista de centroides
    return clusters, centroids

# Função para cálculo da largura média de silhueta
def silhouette_width(X, clusters, centroids):
    # Lista de larguras de silhueta para cada amostra
    S = list()
    # Para cada amostra do dataset
    for i, x in enumerate(X):
        # Determina o cluster da i-ésima amostra
        ja = clusters[i]
        # Calcula a distância entre o centroide do cluster da i-ésima amostra
        # e o demais centroides
        arr = numpy.linalg.norm(centroids[ja] - centroids, axis=1)
        # Remove o elemento 0 do array de distâncias, pois tal elemento é a distância
        # entre o cluster da i-ésima amostra e ele mesmo
        arr[arr == 0.0] = numpy.inf
        # Determina o cluster vizinho mais próximo do cluster da i-ésima amostra
        jb = numpy.argmin(arr)
        # Seleciona as amostras pertencentes ao cluster da i-ésima amostra
        za = numpy.array([x for k, x in enumerate(X) if clusters[k] == ja])
        # Seleciona as amostras pertencentes ao cluster vizinho mais próximo
        zb = numpy.array([x for k, x in enumerate(X) if clusters[k] == jb])
        # Calcula os valores de a e b
        a = numpy.mean(numpy.linalg.norm(x - za, axis=1))
        b = numpy.mean(numpy.linalg.norm(x - zb, axis=1))
        # Calcula a largura de silhueta da amostra
        s = (b - a)/max(a, b)
        S.append(s)
    # Retorna a média das larguras de silhueta obtidas
    return numpy.mean(S)
        
# Se o valor de K não for especificado, calcula as larguras médias de silhueta
# para valores de K no intervalo [2,150]
if args.k is None:
    # Listas de valores de K
    klist = list(range(2, 151))
    # Lista para armazenar silhuetas médias
    slist = list()
    # Percorre a lista de valores de K
    for k in klist:
        # Determina os clusters para cada amostras e seus centroides
        clusters, centroids = kmeans(X, k)
        # Calcula a silhueta média para o agrupamento
        slist.append(silhouette_width(X, clusters, centroids))
    # Construção de uma tabela para exibição dos valores de silhueta média
    # para valores de K de 2 a 10.
    slist_small = slist[:9]
    slist_small = numpy.array(slist_small).reshape(1, len(slist_small))
    ax = seaborn.heatmap(slist_small, cmap='YlGnBu', annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.41})
    ax.set_xlabel('Quantidade de clusters (K)')
    ax.set_xticklabels(['%d' % (k) for k in klist])
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.show()
    # Construção de um gráfico para exibição dos valores de silhueta média
    # para valores de K de 2 a 150.
    ax = seaborn.lineplot(klist, slist)
    ax.set_xlabel('Quantidade de clusters (K)')
    ax.set_ylabel('Largura média de silhueta (S)')
    plt.tight_layout()
    plt.show()
# Se o valor de K for especificado, apenas calcula a largura média de silhueta
# para o valor selecionado
else:
    clusters, centroids = kmeans(X, args.k)
    print(silhouette_width(X, clusters, centroids))