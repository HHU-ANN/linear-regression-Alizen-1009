import numpy as np
x = np.zeros(6)
#x = np.concatenate((np.ones((m,1)),x),axis=1)
_x = np.insert(x, 0, 1)
print(_x)