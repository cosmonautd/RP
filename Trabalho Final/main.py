import os
import sys
import cv2
import time
import numpy
import sklearn.discriminant_analysis
import sklearn.svm
import sklearn.model_selection

dataset = 'rleaf'

X = numpy.genfromtxt(os.path.join('datasets', 'X-%s.csv' % (dataset)), delimiter=',')
Y = numpy.genfromtxt(os.path.join('datasets', 'Y-%s.csv' % (dataset)), delimiter=',')

K = 10
C = len(numpy.unique(Y))

# Instanciação do objeto responsável pela divisão de conjuntos de
# treino e teste de acordo com a metodologia K-Fold com K = 10
cross_val = sklearn.model_selection.StratifiedKFold(K)

# Matriz de confusão
conf_matrix = numpy.zeros((K, C, C))

# Lista de acurácias
accuracies = numpy.zeros(K)

# Percorre as divisões de conjuntos de treino e teste
# 10-Fold
for k, (train_index, test_index) in enumerate(cross_val.split(X,Y)):

    # Assinala os conjuntos de treino e teste de acordo
    # com os índices definidos
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Realiza a inferência
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    # svm = sklearn.svm.SVC(kernel='linear')
    # svm.fit(X_train, Y_train)

    y = lda.predict(X_test)
    # y = svm.predict(X_test)

    # Preenche a matriz de confusão
    for i in range(len(y)):
        conf_matrix[k, int(Y_test[i]), int(y[i])] += 1
    
    # # Impressão da matriz de confusão
    # print('Matriz de confusão %d:' % (k))
    # print(conf_matrix[k])
    
    # Cálculo do número de sucessos usando a matriz de confusão
    # Soma dos elementos da diagonal principal
    success = numpy.sum(numpy.diag(conf_matrix[k]))

    # Cálculo e impressão do resultado da validação
    accuracy = 100*(success/len(y))
    print('%.2f %%' % (accuracy))

    accuracies[k] = accuracy

print('%.2f (+/- %.2f)' % (numpy.mean(accuracies), numpy.std(accuracies)))