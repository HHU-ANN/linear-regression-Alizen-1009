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
    w = np.zeros(n)
    w = w.reshape(-1, 1)
    #print(np.sign(w))
    max_iterator = 100000
    alpha = 1e-10
    lbda = 0.01
    y = y.reshape(y.shape[0], 1)
    #print(y)
    #print(m)
    for i in range(max_iterator):
        gradient = np.dot(x.T,(np.dot(x, w)-y)) / m + lbda * np.sign(w)
        #print(gradient)
        w = w - alpha * gradient 
        #print(w)   
    return w

def lasso(data):
    x, y = read_data()
    weight = lasso_train(x, y)
    data = np.insert(data, 0, 1)
    data = data.reshape(1, data.shape[0])
    #print(data) 
    #print(weight)
    ans = np.dot(data, weight)
    #print(ans)
    return ans[0][0] 

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
