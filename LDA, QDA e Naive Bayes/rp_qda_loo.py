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

def qda(X_train, Y_train, X_test):
    y = list()
    mu = list()
    cov = list()
    for class_ in sorted(list(set(Y_train))):
        idx = numpy.where(Y_train == class_)[0]
        mu.append(numpy.mean(X_train[idx], axis=0))
        cov.append(numpy.cov(X_train[idx], rowvar=False))
    for i, x in enumerate(X_test):
        p = list()
        for j in range(len(mu)):
            p.append(numpy.log(numpy.linalg.det(cov[j])) + numpy.dot(numpy.dot((x - mu[j]).T, numpy.linalg.inv(cov[j])), x - mu[j]) - 2*numpy.log(1/3))
        y_ = numpy.argmin(p)
        y.append(y_)
    return numpy.array(y)

cross_val = LeaveOneOut()
cross_val.get_n_splits(X)

total = len(X)
success = 0.0

for train_index, test_index in cross_val.split(X,Y):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    y = qda(X_train, Y_train, X_test)

    success += sum(y == Y_test)

result = 100*(success/total)
print('%.2f %%' % (result))