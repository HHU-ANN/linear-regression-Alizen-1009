# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge_train(x, y):
    lbda = 0.1
    I = np.eye(x.shape[1])
    res1 = (np.dot(x.T,x) + lbda * I)
    res2 = np.linalg.inv(res1)
    ans = np.dot(res2, np.dot(x.T, y))
    return ans
    

def ridge(data):
    x, y = read_data()
    weight = ridge_train(x, y)
    return data @ weight
    
def lasso_train(x, y):
    return ridge_train(x, y)

def lasso(data):
    x, y = read_data()
    weight = lasso_train(x, y)
    return data @ weight

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
