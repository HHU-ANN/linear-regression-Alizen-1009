import numpy as np
x = np.ones(6)
print(x)
path = "D:/study_file/ANN/linear-regression-Alizen-1009/data/exp02"
# = np.load(path + '/y_train.npy')
#x = np.concatenate((np.ones((m,1)),x),axis=1)
#_x = np.insert(x, 0, 1)
x = x.reshape(-1,1)
print(x)
print(x.shape)