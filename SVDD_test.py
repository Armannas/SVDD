import numpy as np
from matplotlib import pyplot as plt
from SVDD import SVDD
from scipy.linalg import cholesky, LinAlgError
from time import time
import scipy.io as sio
import definitions
import pickle
from sklearn.metrics import confusion_matrix as conf_mat
meani = [0, 0]
covi = [[1, 0], [0, 1]]
Ni = 1000
Xi = np.random.multivariate_normal(meani, covi, Ni)
yi = np.ones(Ni).reshape(Ni, 1)
plt.scatter(Xi[:, 0], Xi[:, 1], c='b')

No = 100
Ro = np.random.uniform(2, 4, No)
theta_o = np.random.uniform(0, np.pi, No)

x_loc = (Ro * np.cos(theta_o)).reshape(No, 1)
y_loc = (Ro * np.sin(theta_o)).reshape(No, 1)

Xo = np.hstack([x_loc, y_loc])
yo = -1 * np.ones(No).reshape(No, 1)

plt.scatter(Xo[:, 0], Xo[:, 1], c='r')

X = np.vstack([Xi, Xo])
y = np.vstack([yi, yo]).ravel()



#%%

Xtrain = np.vstack([X[y == 1][0:900], X[y == -1][0:50]])
ytrain = np.hstack([y[y == 1][0:900], y[y == -1][0:50]])
Xtest = np.vstack([X[y == 1][900:1000], X[y == -1][50:100]])
ytest = np.hstack([y[y == 1][900:1000], y[y == -1][50:100]])

clf = SVDD(kernel_type='rbf', bandwidth=1, fracrej=np.array([0.1, 1]))
clf.fit(Xtrain, ytrain)
y_pred = clf.predict(Xtest)

print(conf_mat(ytest, y_pred, normalize='true'))
p = clf._plot_contour(Xtest, ytest)
p.savefig('test.pdf', bbox_inches='tight',pad_inches = 0)

#%% 3 sample example (all target) and 5 sample example (3 target, 2 outlier)
X1 = np.array([[1, 1], [1.5,2], [2,1]])
y1 = np.array([1, 1, 1])

bw = 1
clf1 = SVDD(kernel_type='rbf', bandwidth=bw)
clf1.fit(X1, y1)
y_pred = clf1.predict(X1)
print(y_pred)


p1 = clf1._plot_contour(X1, y1, [0,3,0,3])

# p.savefig('test.pdf', bbox_inches='tight',pad_inches = 0)
#%%
bw = 0.7
X2 = np.array([[1, 1], [2,2], [2,1], [1.5, 0.8], [1.5, 1] ])
y2 = np.array([1, 1, 1, -1, -1])
clf2 = SVDD(kernel_type='rbf', bandwidth=bw)
clf2.fit(X2, y2)
y_pred = clf2.predict(X2)
p2 = clf2._plot_contour(X2, y2, [0.5,3,0,3])

print(conf_mat(y2, y_pred, normalize='true'))
