import cPickle
import numpy as np
from sklearn.preprocessing import StandardScaler

def one_hot(a, n=10):
    b = np.zeros((len(a),n),dtype = np.float32)
    b[np.arange(len(a)), a] = 1.0
    return b
    

def parse():
    X = cPickle.load(open('mnist.pkl','r'))
    # train, valid, test
    # X, y
    # 50000, 10000, 10000

    print len(X[0][0]), len(X[1][0]), len(X[2][0])

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test)= X 
    #X_train, X_valid, X_test = map(map(lambda x: np.reshape(np.array(x, dtype = np.float32),[28,28])), [X_train, X_valid, X_test])
    X_train, X_valid, X_test = map(np.array, [X_train, X_valid, X_test])
    X_train, X_valid, X_test = map(np.float32, [X_train, X_valid, X_test])

    x1 = X_train[0].reshape(28, 28)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train, X_valid, X_test  = map(scaler.transform, [X_train, X_valid, X_test])

    x2 = X_train[0].reshape(28, 28)

    X_train, X_valid, X_test = map(lambda x: np.reshape(x, [-1, 28, 28] ), [X_train, X_valid, X_test])
    y_train, y_valid, y_test = map(one_hot, [y_train, y_valid, y_test])

    x3 = X_train[0,:,:]
    X = (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    print X_train.shape, X_valid.shape, X_test.shape 
    return X
if __name__=='__main__':
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = parse()
    y = y_train[:12]
    print y
    import matplotlib.pyplot as plt
    plt.imshow(X_train[0])
    plt.show()
