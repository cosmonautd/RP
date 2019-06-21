import os
import sys
import cv2
import time
import numpy
import keras
import logging
import warnings
import tensorflow
import sklearn.cluster

# Supressão de logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore')
tensorflow.get_logger().setLevel(logging.ERROR)

# numpy.set_printoptions(threshold=sys.maxsize)
# numpy.set_printoptions(formatter={'float': lambda x: '%5.2f' % x})

paths = {
    'kimia99': 'datasets/kimia99',
    'fish': 'datasets/fish',
    'leaf': 'datasets/leaf',
    'rleaf': 'datasets/rleaf',
    'sleaf': 'datasets/sleaf',
}

sets = ['leaf']

def zscore(X):
    return (X - numpy.mean(X, axis=0))/numpy.std(X, axis=0, ddof=1)

def readcontour(ctn_path):
    with open(ctn_path) as ctn_file:
        ctn = [list(map(float, line.strip().split())) for line in ctn_file.readlines()]
        ctn = numpy.array(ctn)
    return ctn

def getcontour(image):
    # image = 255 - image # fix for images whose object is black with white background (need to normalize)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1: raise ValueError("Erro: contornos não encontrados")
    main_contour = max(contours, key=lambda x:len(x))
    main_contour = numpy.reshape(main_contour, (len(main_contour), 2))
    main_contour = main_contour.astype(float)
    return main_contour

def mlpcontour(contour, r, q):
    c = numpy.mean(contour, axis=0)
    len_ = len(contour)
    X = numpy.zeros((r*len_,1))
    d = numpy.array(contour)
    D = numpy.linalg.norm(d-c, axis=1)/len_
    counter = 0
    r_ = int(r/2)
    for j in range(len_):
        for k in range(j-r_, j+r_+1, 1):
            if j != k:
                X[counter] = D[k%len_]
                counter += 1
    X = X.reshape((len_, r))
    X = zscore(X)
    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
    mlp = keras.models.Sequential([
        keras.layers.Dense(q, activation='tanh', input_shape=(r,),
                            kernel_initializer=init, bias_initializer=init),
        keras.layers.Dense(1, activation='tanh',
                            kernel_initializer=init, bias_initializer=init)
    ])
    mlp.compile(optimizer='adam', loss='mean_squared_error')
    mlp.fit(X, D, epochs=1, verbose=1)
    W = numpy.array([])
    W = numpy.concatenate((W, mlp.layers[1].get_weights()[0].flatten()), axis=None)
    W = numpy.concatenate((W, mlp.layers[1].get_weights()[1].flatten()), axis=None)
    return W

def rbfcontour(contour, rr, q):
    c = numpy.mean(contour, axis=0)
    len_ = len(contour)
    W = numpy.array([])
    for r in range(2, rr+1, 2):
        X = numpy.zeros((r*len_,1))
        d = numpy.array(contour)
        D = numpy.linalg.norm(d-c, axis=1)/len_
        counter = 0
        r_ = int(r/2)
        for j in range(len_):
            for k in range(j-r_, j+r_+1, 1):
                if j != k:
                    X[counter] = D[k%len_]
                    counter += 1
        X = X.reshape((len_, r))
        X = zscore(X)
        kmeans = sklearn.cluster.KMeans(n_clusters=q, random_state=1).fit(X)
        t = kmeans.cluster_centers_
        Z = list()
        for x in X:
            Z.append(numpy.exp((-numpy.linalg.norm(x-t, axis=1)**2)))
        Z = numpy.array(Z).T
        Z = numpy.concatenate((numpy.ones((1,len_)), Z), axis=0)
        M = numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.pinv(numpy.dot(Z,Z.T))))
        W = numpy.concatenate((W, M), axis=None)
    return W

def lcg_weights(p, q, seed, a, b, c):
    W = list()
    W.append(seed)
    for i in range(q*(p+1)-1):
        r = (a*W[i] + b) % c
        if r not in W:
            W.append(r)
        else:
            S = list(set(list(range(c))) - set(W))
            S.sort()
            W.append(S[int(len(S)/2)])
    W = numpy.array(W)
    W = (W - numpy.mean(W))/numpy.std(W, ddof=1)
    W = W.reshape((q, p+1))
    return W

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def elmcontour(contour, rr, q):
    c = numpy.mean(contour, axis=0)
    len_ = len(contour)
    W = numpy.array([])
    for r in range(2, rr+1, 2):
        X = numpy.zeros((r*len_,1))
        d = numpy.array(contour)
        D = numpy.linalg.norm(d-c, axis=1)/len_
        counter = 0
        r_ = int(r/2)
        for j in range(len_):
            for k in range(j-r_, j+r_+1, 1):
                if j != k:
                    X[counter] = D[k%len_]
                    counter += 1
        X = X.reshape((len_, r))
        X = zscore(X)
        WW = lcg_weights(r, q, q, r, q*(r+1), (q*(r+1))**2)
        X = X.T
        bias_X = -1*numpy.ones(X.shape[1]).reshape(1, X.shape[1])
        X = numpy.concatenate((bias_X, X))
        Z = sigmoid(numpy.dot(WW, X))
        bias_Z = -1*numpy.ones(Z.shape[1]).reshape(1, Z.shape[1])
        Z = numpy.concatenate((bias_Z, Z))
        M = numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.pinv(numpy.dot(Z,Z.T))))
        W = numpy.concatenate((W, M), axis=None)
    return W

for id_ in sets:

    dataset = dict()
    results_table = list()

    files = [f for f in os.listdir(paths[id_])]
    files.sort()
    for f in files:
        if os.path.isfile(os.path.join(paths[id_], f)):
            class_ = f.split('_')[0]
            if class_ not in dataset.keys():
                dataset[class_] = dict()
                dataset[class_]['contours'] = list()
            if f.endswith('.ctn'):
                contour = readcontour(os.path.join(paths[id_], f))
                dataset[class_]['contours'].append(contour)
            elif f.endswith('.png'):
                image = cv2.imread(os.path.join(paths[id_], f), 0)
                contour = getcontour(image)
                dataset[class_]['contours'].append(contour)

    print("Dataset: %s" % (id_))
    print("--")

    results_row = list()

    t_ = 0
    for ii, class_ in enumerate(dataset.keys()):
        dataset[class_]['features'] = list()
        if len(dataset[class_]['contours']) > 0:
            for jj, contour in enumerate(dataset[class_]['contours']):
                checkpoint = time.time()
                print('Class: %d (%s). Shape %d' % (ii, class_, jj))
                features = rbfcontour(contour, 10, 4)
                t_ += time.time() - checkpoint
                dataset[class_]['features'].append(features)
        else:
            print("Erro: lista de contornos vazia")
    
    print('Feature extraction time: %.2f s' % (t_))
    print('Feature extraction time per sample: %.1f ms' % (1000*t_/len(files)))

    X = list()
    Y = list()
    for i, class_ in enumerate(dataset.keys()):
        X += dataset[class_]['features']
        Y += len(dataset[class_]['features'])*[i]

    X = numpy.array(X)
    Y = numpy.array(Y)

    numpy.savetxt('X-%s.csv' % (id_), X, delimiter=",")
    numpy.savetxt('Y-%s.csv' % (id_), Y, delimiter=",")

    print('--')