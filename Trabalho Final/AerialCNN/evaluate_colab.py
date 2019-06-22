import os
from google.colab import drive

drive.mount('/content/drive')

# Caminho das imagens e seus rótulos
X_path = 'drive/My Drive/Colab Notebooks/AerialCNN/dataset/X'
Y_path = 'drive/My Drive/Colab Notebooks/AerialCNN/dataset/Y'

# Criação de diretório para armazenar os pesos treinados
weights_path = 'drive/My Drive/Colab Notebooks/AerialCNN/weights'
if not os.path.exists(weights_path):
    os.mkdir(weights_path)

import os
import cv2
import logging
import warnings
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

def debug(vars):
    for var in vars: print(var)
    quit()

# Definição do nível de log do Python e TensorFlow
tensorflow.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

# Função para plot dos gráficos de perda durante o treinamento
def visualize_training(hist):
    # Plot da perda
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

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
    plt.show()

# Definição do tipo de dado Dataset
Dataset = collections.namedtuple('Dataset', 'x_train y_train x_valid y_valid x_test y_test')

X = []
Y = []
for imagename in sorted(os.listdir(X_path)):
    if imagename.endswith('.jpg'):
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

    # Aumento de dados
    datagen_args = dict(rotation_range=360, width_shift_range=0.3, height_shift_range=0.3,
                        shear_range=30, zoom_range=0.2, fill_mode='reflect', 
                        horizontal_flip=True, vertical_flip=True)

    datagen_train_x = ImageDataGenerator(**datagen_args)
    datagen_train_y = ImageDataGenerator(**datagen_args)
    datagen_val_x = ImageDataGenerator(**datagen_args)
    datagen_val_y = ImageDataGenerator(**datagen_args)

    datagen_seed = 1
    datagen_train_x.fit(X_train, augment=True, seed=datagen_seed)
    datagen_train_y.fit(Y_train, augment=True, seed=datagen_seed)
    datagen_val_x.fit(X_val, augment=True, seed=datagen_seed)
    datagen_val_y.fit(Y_val, augment=True, seed=datagen_seed)

    xtraingen = datagen_train_x.flow(X_train, seed=datagen_seed, batch_size=32)
    ytraingen = datagen_train_y.flow(Y_train, seed=datagen_seed, batch_size=32)
    xvalgen = datagen_val_x.flow(X_val, seed=datagen_seed, batch_size=32)
    yvalgen = datagen_val_y.flow(Y_val, seed=datagen_seed, batch_size=32)

    # Definição do caminho para salvamento dos pesos
    weights_path_loo = os.path.join(weights_path, 'weights_%d.hdf5' % (test_index[0]))
    
    # Definição de um callback para salvamento da melhor combinação de pesos
    checkpoint = ModelCheckpoint(weights_path_loo, verbose=1, save_best_only=True)

    dims = X_train.shape[1:]

    aerialcnn = aerialcnn_model(dims)
    callbacks = [checkpoint, EarlyStopping(min_delta=0.001, patience=20)]
    history = aerialcnn.fit_generator(zip(xtraingen, ytraingen), epochs=100, callbacks=callbacks,
                                      steps_per_epoch=4*len(X_train), validation_data=zip(xvalgen, yvalgen),
                                      validation_steps=4*len(X_val))
    
    # Plot dos gráficos de acurácia e perda
    visualize_training(history)

    # Carregamento da melhor combinação de pesos obtida durante o treinamento
    aerialcnn.load_weights(weights_path_loo)

    # Teste
    y = aerialcnn.predict(X_test)
    show_image([X[test_index][0], np.squeeze(y[0])])
