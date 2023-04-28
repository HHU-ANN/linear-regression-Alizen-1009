# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge_train(x, y):
    lbda = 0.03
    m = x.shape[0]
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    I = np.eye(x.shape[1])
    res = np.dot(x.T,x) + lbda * I
    res_inv = np.linalg.inv(res)
    w = np.dot(res_inv, np.dot(x.T, y))
    return w

def ridge(data):
    x, y = read_data()
    weight = ridge_train(x, y)
    data = np.insert(data, 0, 1)
    return data @ weight + 0.5


def lasso_train(x, y):
    m = x.shape[0]
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    n = x.shape[1]
    w = np.ones(n)
    max_iterator = 1000
    alpha = 0.1
    lbda = 0.03
    y = y.reshape(1, -1)
    for i in range(max_iterator):
        gradient = np.dot(x.T,(np.dot(x, w)-y)) / m + lbda * np.sign(w)
        w = w - alpha * gradient
    return w

def lasso(data):
    x, y = read_data()
    weight = lasso_train(x, y)
    data = np.insert(data, 0, 1)
    return data @ weight + 0.5

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
