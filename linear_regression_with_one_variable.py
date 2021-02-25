"linear regression with one varialble"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =  pd.read_csv(,header = None)   "Enter the data file path or name"

"""To have a basic knowledge of data set we must look at the data's few example and
description like count,mean,standard deviation, max ,min etc"""

data.head() "for first few examples"
data.describe()

"plot of data for visualization"

plt.scatter(data[0],data[1]) "for scattering plot of data"
plt.xticks(np.arange(5,30,step = 5)) "markings on x axis"
plt.yticks(np.arange(-5,30,step = 5)) "markings on y axis"
plt.xlabel("") "Enter x label"
plt.ylabel("") "Enter y label"
plt.title("")"Enter title"

"Computing cost function,J(theta)"

def J_theta(X,y,theta) "X,y,theta will be numpy array"
    m = len(y) "number of total y's"
    hypothesis  = X.dot(theta)
    cost_function = (hypothesis - y)**2

    return (1/(2*m) * np.sum(cost_function))

"""In the function we have defined the variables, now we have to initialise the variables
,we can use the same variable names and the place those initalised X,y,theta into J(theta)function"""

data_n = data.values
m = len(data_n[:-1])
X = np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis = 1)
y = data_n.[:-1],reshape(m,1)
theta = np.zeroes((2,1))

J_theta(X,y,theta)

"Gradient Descent"
def gradient_descent(X,y,theta,alpha,iterations) "aplha(learning rate),iterations(number of times to converge)"
    m = len(y)
    J_history = []
    for i in range(iterations):
        hypothesis = X.dot(theta)
        cost_function = np.dot(X.transpose(),(hypothesis - y))
        descent = alpha * 1/m * cost_function
        theta = theta - descent
        J_history.append(J_theta(X,y,theta))

    return  theta,J_history

theta , J_history = gradient_descent(X,y,theta,,) "choose alpha and iterations"

print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1


from mpl_toolkits.mplot3d import axes3d

theta_0vals = np.linspace() "a/c to my resourse (-10,10,100)"
theta_1vals = np.linspace() "a/c to my resourse (-1,4,100)"

J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta_0vals)):
    for j in range(len(theta_1vals)):
        t=np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i,j] = J_theta(X,y,t)

"For plotting and visualising thr graph"

#Generating the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap="coolwarm")
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel()
plt.ylabel()
plt.title()

"Defining a function hypothesis"
def hypothesis(x,theta)
    hypothesis = np.dot(theta.transpose(),x)
    return hypothesis[0]
hypothesis1 = hypothesis(np.array([1,3.5]),theta)*10000
print(hypothesis1)"write a presentable value for hypothesis1"
