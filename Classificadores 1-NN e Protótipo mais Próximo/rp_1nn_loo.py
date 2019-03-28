import time
import numpy
from sklearn.model_selection import LeaveOneOut

samples = list()
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)
X = dataset[:,0:len(dataset[0])-3]
Y = dataset[:,len(dataset[0])-3:]
Y = numpy.argmax(Y, axis=1)

def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

X = zscore(X)

t_train = [0]
t_classification = list()
def one_nn(X_train, Y_train, X_test):
    y = list()
    for x in X_test:
        checkpoint = time.time()
        dist = numpy.linalg.norm(X_train - x, axis=1)
        y_ = Y_train[numpy.argmin(dist)]
        t_classification.append(time.time() - checkpoint)
        y.append(y_)
    return numpy.array(y)

cross_val = LeaveOneOut()
cross_val.get_n_splits(X)

total = len(X)
success = 0.0

for train_index, test_index in cross_val.split(X,Y):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    y = one_nn(X_train, Y_train, X_test)

    success += sum(y == Y_test)

result = 100*(success/total)
print('%.2f %%' % (result))

print('Tempo médio de treinamento: %f ms' % (1000*numpy.mean(t_train)))
print('Tempo médio de classificação: %f ms' % (1000*numpy.mean(t_classification)))