import numpy as np
x = np.zeros((6,6))
m = x.shape[0]
x = np.concatenate((np.ones((m,1)),x),axis=1)
print(x)
print(x.shape)