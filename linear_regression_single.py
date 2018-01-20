# this is a single variable linear regression model
# it predicts the number of expected bike rentals based on the temperature for that day

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datacleaner
from compute_cost import compute_cost
from gradient_descent import gradient_descent

# read data
data = pd.read_csv('dataset_linear.csv')
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:1])
y = np.matrix(clean_data[:, 1:2])

# plot the data
plt.plot(X[:, 0], y, 'ro')

# add ones
X0 = np.ones((X.shape[0], 1))
X = np.hstack((X0, X))

# initialize thetas
theta = np.zeros((X.shape[1], 1))

# initialize learning parameters
iterations = 1000
alpha = 1.4

# test cost function
J = compute_cost(X, y, theta)

# train thetas to fit data
[theta, J_history] = gradient_descent(X, y, theta, alpha, iterations)

# plot hypothesis on top scatter of data
x_init = np.matrix(np.arange(0, 1, 0.01)).T
x0 = np.ones((x_init.size, 1))
x = np.hstack((x0, x_init))
h = x.dot(theta)
plt.cla()
plt.xlim(0, x.max())
plt.scatter(clean_data[:, 0:1], clean_data[:, 1:2])
plt.plot(x_init, h, 'red')
