
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import random


np.random.seed(2)

means = [[2, 2, 3], [4, 2, 3]]
cov = [[.3, .2, .4], [.2, .3, .5], [.2, .3, .6]]
N = 90
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)


def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    
    return np.array_equal(h(w, X), y) #True if h(w, X) == y else False

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(4, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 

                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)
print(m)
print(w[-1])
# print(len(w))
def fun(x, y, w):
    w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
    return -(x*w1+y*w2+w0)/w3

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0, 6.0, 0.1)
X, Y = np.meshgrid(x, y)
zs = np.array(fun(np.ravel(X), np.ravel(Y),w[-1]))
Z = zs.reshape(X.shape)
ax.scatter(X0[0,:], X0[1,:],X0[2,:],marker = "^", color = "red")
ax.scatter(X1[0,:], X1[1,:],X1[2,:],marker = "o", color ="black")
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()