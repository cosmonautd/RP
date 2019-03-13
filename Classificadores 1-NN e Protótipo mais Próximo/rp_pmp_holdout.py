import numpy
from sklearn.model_selection import StratifiedShuffleSplit

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

def pmp(X_train, Y_train, X_test):
    y = list()
    centroids = list()
    for class_ in sorted(list(set(Y_train))):
        idx = numpy.where(Y_train == class_)[0]
        centroids.append(numpy.mean(X_train[idx], axis=0))
    for x in X_test:
        dist = numpy.linalg.norm(centroids - x, axis=1)
        y_ = numpy.argmin(dist)
        y.append(y_)
    return numpy.array(y)

cross_val = StratifiedShuffleSplit(n_splits=20, test_size=0.3)
cross_val.get_n_splits(X)

success = 0.0

for train_index, test_index in cross_val.split(X,Y):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    y = pmp(X_train, Y_train, X_test)

    success += sum(y == Y_test)/len(Y_test)

result = 100*(success/20)
print('%.2f %%' % (result))