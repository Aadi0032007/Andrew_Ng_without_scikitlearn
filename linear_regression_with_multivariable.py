"Linear regression with multi variable"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = os.getcwd() "current directory of file in process"
data = pd.csv_read (path,header = None , names = []) "Enter name according to the requirements"
data = (data - data.mean())/data.std() "feature scaling"
data.head() "For first five data example"
data.describe()

"Cost function, J(theta)"
def J_theta(X,y,theta) "X,y,theta will be numpy array"
    m = len(y) "number of total y's"
    hypothesis  = X.dot(theta)
    cost_function = (hypothesis - y)**2

    return (1/(2*m) * np.sum(cost_function))

"gradient descent algo"

def gradient_descent(X,y,theta,alpha,iterations)
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = J_theta(X, y, theta)

    return theta, cost

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))


g, cost = gradientDescent(X, y, theta, alphxa, iters)

J_theta(X, y, g)


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')

