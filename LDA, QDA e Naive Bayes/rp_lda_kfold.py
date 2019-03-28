import time
import numpy
from sklearn.model_selection import StratifiedKFold

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

t_train = list()
t_classification = list()
def lda(X_train, Y_train, X_test):
    y = list()
    mu = list()
    checkpoint = time.time()
    cov = numpy.cov(X_train, rowvar=False)
    cov_det = numpy.linalg.det(cov)
    cov_inv = numpy.linalg.inv(cov)
    for class_ in sorted(list(set(Y_train))):
        idx = numpy.where(Y_train == class_)[0]
        mu.append(numpy.mean(X_train[idx], axis=0))
    t_train.append(time.time() - checkpoint)
    for x in X_test:
        checkpoint = time.time()
        p = list()
        for j in range(len(mu)):
            # p.append(numpy.log(numpy.linalg.det(cov)) + numpy.dot(numpy.dot((x - mu[j]).T, numpy.linalg.inv(cov)), x - mu[j]) - 2*numpy.log(1/3))
            p.append( (1/numpy.sqrt(2*numpy.pi*cov_det)) * numpy.exp( -0.5*numpy.dot(numpy.dot((x - mu[j]).T, cov_inv), x - mu[j]) ) * (1/3))
        # y_ = numpy.argmin(p)
        y_ = numpy.argmax(p)
        t_classification.append(time.time() - checkpoint)
        y.append(y_)
    return numpy.array(y)

cross_val = StratifiedKFold(10)
cross_val.get_n_splits(X)

total = len(X)
success = 0.0

for train_index, test_index in cross_val.split(X,Y):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    y = lda(X_train, Y_train, X_test)

    success += sum(y == Y_test)

result = 100*(success/total)
print('%.2f %%' % (result))

print('Tempo médio de treinamento: %f us' % (10**6*numpy.mean(t_train)))
print('Tempo médio de classificação: %f us' % (10**6*numpy.mean(t_classification)))