# this is a linear regression model utilizing a quadratic approach to better fit the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datacleaner
from compute_cost import compute_cost
from gradient_descent import gradient_descent

# extract data
data = pd.read_csv('dataset_quadratic.csv')
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:1])
y = np.matrix(clean_data[:, 1:2])

# divide X by 10 to prevent overflow from squared error
X = X / 10

# get no. of training sets]
m = y.shape[0]

# preprocess y to convert 0 hour to 24
for i in range(0, m):
    X[i] = 2.4 if X[i] == 0 else X[i]

# add ones
X0 = np.ones((X.shape[0], 1))
X2 = np.square(X)
X3 = np.power(X, 3)
X = np.hstack((X0, X, X2, X3))

# initialize thetas
theta = np.zeros((X.shape[1], 1))

# initialize learning parameters
iterations = 1000
alpha = 0.01

# test cost function
J = compute_cost(X, y, theta)

# train model through gradient descent
[theta, J_history] = gradient_descent(X, y, theta, alpha, iterations)

# plot J_history to make sure gradient descent worked
plt.figure(2)
plt.plot(range(iterations), J_history)
plt.show()

# plot hypothesis function
x_init = np.matrix(np.arange(0, 24)).T
x_init = x_init / 10
x0 = np.ones((x_init.shape[0], 1))
x2 = np.square(x_init)
x3 = np.power(x_init, 3)
x = np.hstack((x0, x_init, x2, x3))
h = x.dot(theta)
plt.cla()
plt.xlim(0, 2.4)
plt.scatter(clean_data[:, 0:1] / 10, clean_data[:, 1:2])
plt.plot(x_init, h, 'red')
