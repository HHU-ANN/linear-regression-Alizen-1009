from Linear_Regression import lasso
import numpy as np

x = lasso(np.load('../data/exp02/X_train.npy'))

print(x)