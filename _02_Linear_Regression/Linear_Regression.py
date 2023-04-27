# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge_train(x, y):
    lbda = 30
    m = x.shape[0]
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    I = np.eye(x.shape[1])
    res = np.dot(x.T,x) + lbda * I
    res_inv = np.linalg.inv(res)
    ans = np.dot(res_inv, np.dot(x.T, y))
    return ans

def ridge(data):
    x, y = read_data()
    weight = ridge_train(x, y)
    data = np.insert(data, 0, 1)
    return data @ weight



def lasso_train(x, y):
    m = x.shape[0]
    n = x.shape[1]
    W=np.ones((n,1))
    y.reshape(-1,1)
    Lambda = 0.1
    a = 0.01
    for i in range(10000):
        gradient = np.dot(x.T,(np.dot(x, W) -y)/m )+ Lambda * np.sign(W)
        W=W-a* gradient
    return W

def lasso(data):
    x, y = read_data()
    weight = ridge_train(x, y)
    return data @ weight

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
