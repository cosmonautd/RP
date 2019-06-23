import os
import cv2
import logging
import warnings
import itertools
import tensorflow
import collections
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection

# Importação de modelos, camadas, datasets e utilidades do Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10, fashion_mnist

# Importação de otimizações para a CNN
from keras.layers import BatchNormalization
from keras.optimizers import rmsprop
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

import graphmapx

def debug(vars):
    for var in vars: print(var)
    quit()

# Definição do nível de log do Python e TensorFlow
tensorflow.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Caminho das imagens e seus rótulos
X_path = 'dataset/X'
Y_path = 'dataset/Y'
KP_path = 'dataset/kp'
KN_path = 'dataset/kn'

# Diretório para armazenar os pesos treinados
weights_path = 'weights'

# Diretório para armazenar os pesos treinados
paths_path = 'paths'

# Diretório para as saídas do teste
output_path__ = 'output'
if not os.path.exists(output_path__):
    os.mkdir(output_path__)

# Função para construção da AerialCNN
def aerialcnn_model(dims):
    weight_decay = 1e-4
    # Organização sequencial de camadas
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(weight_decay), 
                     input_shape=dims))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(8, kernel_size=3, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(1, kernel_size=3, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))

    # Compilação do modelo. Definição da função de perda e algoritmo de treinamento.
    optimized_rmsprop = rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss="mean_squared_error", optimizer=optimized_rmsprop)
    return model

def show_image(images):
    n = len(images)
    if n == 1:
        fig, (ax0) = plt.subplots(ncols=1)
        ax0.imshow(images[0], cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_visible(False)
    else:
        fig, axes = plt.subplots(ncols=n, figsize=(4*n, 4))
        for ax, image in zip(axes, images):
            ax.imshow(image, cmap='gray', interpolation='bicubic')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()

def image_histogram(image, channels, color='k'):
    fig, (ax0) = plt.subplots(ncols=1)
    hist = cv2.calcHist([image], channels, None, [100], [0, 1])
    ax0.plot(np.arange(0, 1, 1/100), hist, color=color)
    fig.tight_layout()

def load_image(path):
    """ Loads image from path, converts to RGB
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def grid_list(image, r):
    """ Returns a list of square coordinates representing a grid over image
        Every square has length and height equals to r
    """
    height, width, _ = image.shape
    # assertions that guarantee the square grid contains all pixels
    assert r > 0, "Parameter r must be larger than zero"
    if (height/r).is_integer() and (width/r).is_integer():
        glist = []
        for toplefty in range(0, height, r):
            for topleftx in range(0, width, r):
                glist.append((topleftx, toplefty, r))
        return glist
    else:
        new_height = int(r*np.floor(height/r))
        new_width = int(r*np.floor(width/r))
        if new_height > 0 and new_width > 0:
            y_edge = int((height - new_height)/2)
            x_edge = int((width - new_width)/2)
            glist = []
            for toplefty in range(y_edge, y_edge+new_height, r):
                for topleftx in range(x_edge, x_edge+new_width, r):
                    glist.append((topleftx, toplefty, r))
            return glist
        else:
            raise ValueError("r probably larger than image dimensions")

def score(path, ground_truth, r):
    score_ = 1.0
    penalty = 0.03
    T = list()
    for px in path:
        # t.append(np.mean(ground_truth[px[0], px[1]])/255)
        h, w, _ = ground_truth.shape
        a = max(0, px[0]-int(r/2))
        b = min(h-1, px[0]+int(r/2))
        c = max(0, px[1]-int(r/2))
        d = min(w-1, px[1]+int(r/2))
        t = ground_truth[a:b, c:d]
        t = t.mean(axis=2)/255
        t = cv2.erode(t, np.ones((int(r/2), int(r/2)), np.uint8), iterations=1)
        T.append(np.mean(t))
    for i, t in enumerate(T):
        if i < len(T) - 1 and i > 0:
            if t < 0.5: score_ = np.maximum(0, score_ - penalty*(1-t))
    return score_

def draw_path(image, path, color=(0,255,0), found=False):
    image_copy = image.copy()
    if len(image_copy.shape) < 3:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
    centers = [(p[0], p[1]) for p in path]
    cv2.circle(image_copy, centers[0][::-1], 6, color, -1)
    cv2.circle(image_copy, centers[-1][::-1], 12, color, -1)
    if found:
        for k in range(len(centers)-1):
            r0, c0 = centers[k]
            r1, c1 = centers[k+1]
            cv2.line(image_copy, (c0, r0), (c1, r1), color, 5)
        # r0, c0 = int(numpy.mean([center[0] for center in centers[-2:]])), int(numpy.mean([center[1] for center in centers[-2:]]))
        # r1, c1 = centers[-1]
        # cv2.arrowedLine(image_copy, (c0, r0), (c1, r1), color, 5, 2, 0, 1)
    return image_copy

def save_image(path, images):
    n = len(images)
    if n == 1:
        fig, (ax0) = plt.subplots(ncols=1)
        ax0.imshow(images[0], cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_visible(False)
    else:
        fig, axes = plt.subplots(ncols=n, figsize=(4*n, 4))
        for ax, image in zip(axes, images):
            ax.imshow(image, cmap='gray', interpolation='bicubic')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Definição do tipo de dado Dataset
Dataset = collections.namedtuple('Dataset', 'x_train y_train x_valid y_valid x_test y_test')

# Caminho das imagens e seus rótulos
X_path = 'dataset/X'
Y_path = 'dataset/Y'

X = []
Y = []
for imagename in sorted(os.listdir(X_path)):
    x = cv2.imread(os.path.join(X_path, imagename))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    y = cv2.imread(os.path.join(Y_path, imagename), cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (125, 125))/np.max(y)
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.expand_dims(np.array(Y), 3)

# LeaveOneOut
cross_val = sklearn.model_selection.LeaveOneOut()
cross_val.get_n_splits(X)

# Total de amostras no dataset original
total = len(X)

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X):

    # Índice do conjunto de validação
    val_index = (test_index + 1) % total
    train_index = np.array([x for x in train_index if x not in val_index])

    # Assinala os conjuntos de treino, validação e teste
    X_train, X_test, X_val = X[train_index], X[test_index], X[val_index]
    Y_train, Y_test, Y_val = Y[train_index], Y[test_index], Y[val_index]

    dims = X_train.shape[1:]

    aerialcnn = aerialcnn_model(dims)

    # Definição do caminho para salvamento dos pesos
    weights_path_loo = os.path.join(weights_path, 'weights_%d.hdf5' % (test_index[0]))

    # Carregamento da melhor combinação de pesos obtida durante o treinamento
    aerialcnn.load_weights(weights_path_loo)

    # Calcular perda sobre conjunto de teste
    print('Test loss: %.4f' % (aerialcnn.evaluate(X_test, Y_test, verbose=0)))

    # Teste
    y = aerialcnn.predict(X_test)
    y = np.squeeze(y[0])

    y__ = np.squeeze(Y_test[0])

    print("MSE 0: %.4f" % (((y - y__)**2).mean(axis=None)))

    # y = (y - np.min(y))
    # y = y/np.max(y)

    # print("MSE 1: %.4f" % (((y - y__)**2).mean(axis=None)))

    # cv2.imwrite(os.path.join(output_path__, 'tfcn-output-%02d.jpg' % (test_index[0]+1)), 255*y)

    # show_image([X[test_index][0], y__, y])
    # plt.show()

    # continue

    # Histograma
    # show_image([X[test_index][0], y])
    # image_histogram(y, [0])

    # Exibição dos gráficos
    # plt.show()

    # Pós-processamento
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    # y = np.array(clahe.apply((255*y).astype(np.uint8)), dtype=np.float32)/255
    # show_image([X[test_index][0], y])

    t_matrix = y
    ground_truth = load_image(os.path.join(Y_path, 'aerial%02d.jpg' % (test_index[0]+1)))

    keypoints_image = load_image(os.path.join(KP_path, 'aerial%02d.jpg' % (test_index[0]+1)))
    grid = grid_list(X_test[0], 8)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    th = cv2.calcHist(y, [0], None, [100], [0, 1])
    th = th.flatten()
    tv = np.arange(0, 1, 1/100)
    c = np.sum(th*tv)/np.sum(th)

    router = graphmapx.RouteEstimator(r=8, c=c, grid=grid)
    G = router.tm2graph(t_matrix)

    output_path = 'paths/%s' % ('aerial%02d' % (test_index[0]+1))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t, t_matrix)

        score__ = score(path, ground_truth, 8)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText    = (10, 40)
        fontScale              = 1
        lineType               = 2

        fontColor = (255,0,0) if score__ < 0.7 else (0,255,0)

        path_image = draw_path(X_test[0], path, found=found, color=fontColor)
        cv2.putText(path_image, 'Score: %.2f' % (score__),
                    topLeftCornerOfText, font, fontScale, fontColor, lineType)

        save_image(os.path.join(output_path, 'path-%d.jpg' % (counter+1)), [path_image])
