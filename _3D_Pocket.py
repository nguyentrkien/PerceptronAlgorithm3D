
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
from sklearn import datasets
import random
np.random.seed(2)
N = 40
X, Y = datasets.make_blobs(n_samples = N, n_features = 3, centers = 2, cluster_std = 2.2, random_state = 2)
X0 = X[:,:][Y==0].T
X1 = X[:,:][Y==1].T
X = np.concatenate((X0, X1), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, N)), X), axis = 0)
def Convert(x):
  if x >= 0: return 1
  else: return 0

def Convergence(x):
    for i in range(N):       
        x[i] = Convert(x[i])        
    return x

def Count(Dot_Product, y):
    count = 0
    for m, n in zip(Dot_Product, y):       
        if m != n:
            count += 1
    return count

def perceptron(X, Y, w_init, Loop):
    w = [w_init]
    N = X.shape[1]  
    miss_point =[]
    X_temp = X
    pocket = [w.copy()]
    misclassified = []
    w = np.array(w).reshape(4,1) 
    for Loop in range(Loop):
        for i in range(N):  
            
            if (Y[i]==1) and (np.dot(X_temp[:,i], w)) < 0:
                    w = w + X_temp[:,i].T
            if (Y[i]==0) and (np.dot(X_temp[:,i], w)) >= 0:
                    w = w - X_temp[:,i].T
            #Append old pocket to new w    
            pocket.append(w.copy()) 
            w = np.array(w).reshape(4,) 
           
            Dot_Product = Convergence(np.dot(w.T, X_temp)).tolist()  
            
            misclassified.append(Count(Dot_Product, np.sort(Y, axis = None)))
            #print(Count(Dot_Product, Y),Dot_Product)
    index_min = misclassified.index(min(misclassified))
    return (pocket[index_min+1], misclassified[index_min])

d = X.shape[0]
w_init = np.random.randn(d, 1)
#Loop max = 150
w, m = perceptron(X, Y, w_init, 150)
print(w, m)
print(Y)
print(Convergence(np.dot(w.T, X)))
print(np.dot(w.T, X))

def fun(x, y, w):
    w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
    return [-(x*w1+y*w2+w0)]/w3

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-10, 10.0, 0.1)
X, Y = np.meshgrid(x, y)
zs = np.array(fun(np.ravel(X), np.ravel(Y),w))
Z = zs.reshape(X.shape)
plt.title(f"Pocket Algorithm\nw = {w}\nMiss classifies point: {format(m)}/{format(X.shape[0])}"+ "\n" + f"Accuracy of Pocket Algorithm is {(X.shape[0] - m)*100/X.shape[0]}" + "%")
ax.scatter(X0[0,:], X0[1,:],X0[2,:],marker = "^", color = "red")
ax.scatter(X1[0,:], X1[1,:],X1[2,:],marker = "o", color ="black")
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

