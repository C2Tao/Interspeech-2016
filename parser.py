import cPickle
import numpy as np
from sklearn.preprocessing import StandardScaler

X = cPickle.load(open('mnist.pkl','r'))
# train, valid, test
# X, y
# 50000, 10000, 10000

print len(X[0][0]), len(X[1][0]), len(X[2][0])

(X_train, y_train), (X_valid, y_valid), (X_test, y_test)= X 

X_train, X_valid, X_test  = map(np.array, [X_train, X_valid, X_test])
X_train, X_valid, X_test  = map(np.float32, [X_train, X_valid, X_test])

x1 = X_train[0].reshape(28, 28)

scaler = StandardScaler()
scaler.fit(X_train)

X_train, X_valid, X_test  = map(scaler.transform, [X_train, X_valid, X_test])

x2 = X_train[0].reshape(28, 28)

X = (X_train, y_train), (X_valid, y_valid), (X_test, y_test) 

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.imshow(np.concatenate([x1,x2]))
    print x2
    plt.show()
