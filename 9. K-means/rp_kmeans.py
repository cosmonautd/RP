import time
import numpy
import argparse
import random

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, default=3, help="Número de sementes")
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

def kmeans(X, k):
    N, d = X.shape
    centroids = numpy.array(random.sample(list(X), k))
    clusters = numpy.zeros(N).astype(int)
    while True:
        for i, x in enumerate(X):
            dist = numpy.linalg.norm(x - centroids, axis=1)
            m = numpy.argmin(dist)
            clusters[i] = m
        centroids_ = numpy.copy(centroids)
        for j, _ in enumerate(centroids):
            z = numpy.array([x for i, x in enumerate(X) if clusters[i] == j])
            if len(z) > 0:
                centroids[j] = numpy.mean(z, axis=0)
        if numpy.all(centroids == centroids_): break
    return clusters, centroids

def silhouette_width(X, clusters, centroids):
    S = list()
    for i, x in enumerate(X):
        ja = clusters[i]
        arr = numpy.linalg.norm(centroids[ja] - centroids, axis=1)
        arr[arr == 0.0] = numpy.inf
        jb = numpy.argmin(arr)
        za = numpy.array([x for k, x in enumerate(X) if clusters[k] == ja])
        zb = numpy.array([x for k, x in enumerate(X) if clusters[k] == jb])
        a = numpy.mean(numpy.linalg.norm(x - za, axis=1))
        b = numpy.mean(numpy.linalg.norm(x - zb, axis=1))
        s = (b - a)/max(a, b)
        S.append(s)
    return numpy.mean(S)
        

clusters, centroids = kmeans(X, args.k)
quality = silhouette_width(X, clusters, centroids)
print(quality)