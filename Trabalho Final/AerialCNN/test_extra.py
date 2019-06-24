import os
import cv2
import time
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
X_path = 'dataset/extra/X'
Y_path = 'dataset/extra/Y'
KP_path = 'dataset/extra/kp'
KN_path = 'dataset/extra/kn'

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

for imagename in sorted(os.listdir(X_path)):

    if imagename != 'suburbs1000.jpg': continue

    x = cv2.imread(os.path.join(X_path, imagename))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(x, axis=0)

    dims = x.shape[1:]

    aerialcnn = aerialcnn_model(dims)

    # Definição do caminho para salvamento dos pesos
    weights_path_loo = os.path.join(weights_path, 'weights_%d.hdf5' % (0))

    # Carregamento da melhor combinação de pesos obtida durante o treinamento
    aerialcnn.load_weights(weights_path_loo)

    # Teste
    time__ = time.time()

    y = aerialcnn.predict([x])
    y = np.squeeze(y[0])

    print('Time: %.3f s' % (time.time() - time__))

    # y = (y - np.min(y))
    # y = y/np.max(y)

    cv2.imwrite(os.path.join(output_path__, 'tfcn-output-%s.jpg' % (imagename[:-4])), 255*y)

    # show_image([X[test_index][0], y])
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

    r__ = 8

    t_matrix = y

    keypoints_image = load_image(os.path.join(KP_path, imagename))
    grid = grid_list(np.squeeze(x), r__)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    th = cv2.calcHist(y, [0], None, [100], [0, 1])
    th = th.flatten()
    tv = np.arange(0, 1, 1/100)
    c = np.sum(th*tv)/np.sum(th)

    router = graphmapx.RouteEstimator(r=r__, c=c, grid=grid)
    G = router.tm2graph(t_matrix)

    output_path = 'paths/%s' % (imagename[:-4])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t, t_matrix)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText    = (10, 40)
        fontScale              = 1
        lineType               = 2

        fontColor = (0,255,0)

        path_image = draw_path(np.squeeze(x), path, found=found, color=fontColor)

        save_image(os.path.join(output_path, 'path-%d.jpg' % (counter+1)), [path_image])
