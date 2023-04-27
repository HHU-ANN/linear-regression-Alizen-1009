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
    res = np.dot(x.T,x) + lbda * I
    res_inv = np.linalg.inv(res)
    ans = np.dot(res_inv, np.dot(x.T, y))
    return ans


def ridge(data):
    x, y = read_data()
    weight = ridge_train(x, y)
    return data @ weight
    
def lasso_train(x, y):
    n, m = x.shape[1], x.shape[0]
    p_lambda = 2 * n  * 0.01
    w = np.random.rand(n)
    def loss(w):
        delta_x = y - np.matmul(x,w)
        return np.matmul(delta_x.T,delta_x)/m + p_lambda/n*np.sum(np.abs(w))
    cnt = 0
    optimize_cnt = 0
    while True:
        choice_i = cnt%n
        else_i = [i for i in range(n) if i != choice_i]
        cnt+=1
        else_delta_y = y-np.matmul(x[:,else_i],w[else_i].reshape([-1,1])).reshape(-1)
        D = p_lambda/n
        C = np.sum(np.square(x[:,choice_i]))/m
        B = -2/m * np.sum(np.multiply(x[:,choice_i],else_delta_y) )
        A = p_lambda/n*np.sum(np.abs(w[else_i])) + 1/m*np.sum(np.square(else_delta_y))
        w_i_new_list = [0]
        if -(B+D)/(2*C) > 0:
            w_i_new_list.append(-(B + D) / (2 * C))
        if -(B-D)/(2*C) < 0:
            w_i_new_list.append(-(B - D) / (2 * C))
        old_loss = loss(w)
        new_loss_list = []
        for i in range(3):
            w[choice_i] = w_i_new_list[i]
            new_loss_list.append(loss(w))
        w[choice_i] = w_i_new_list[int(np.argmin(new_loss_list))]
        new_loss = loss(w)
        if old_loss-new_loss<1e-10:
            optimize_cnt += 1
        else:
            optimize_cnt= 0
        if optimize_cnt>n:
            break
    return w


def lasso(data):
    x, y = read_data()
    weight = lasso_train(x, y)
    return data @ weight

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
