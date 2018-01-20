# this is a multivariate linear regression model
# it predicts the number of expected bike rentals based on the temperature, humidity and windspeed for that day

# NOTE: this model still has a fairly large cost but that is because there is a weak correlation between the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datacleaner
from compute_cost import compute_cost
from gradient_descent import gradient_descent

# import the data
data = pd.read_csv('dataset_multi.csv')
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:3])
y = np.matrix(clean_data[:, 3:4])

# plot the data
plt.figure(1)
plt.plot(X[:, 0], y, 'ro')
plt.plot(X[:, 1], y, 'yo')
plt.plot(X[:, 2], y, 'bo')
plt.show()

# add ones
X0 = np.ones((X.shape[0], 1))
X = np.hstack((X0, X))

# initialize thetas
theta = np.zeros((X.shape[1], 1))

# initialize learning parameters
alpha = 0.1
iterations = 100

# test cost function
J = compute_cost(X, y, theta)

# run gradient descent
[theta, J_history] = gradient_descent(X, y, theta, alpha, iterations)

# plot J_history to check that gradient descent worked
plt.figure(2)
plt.plot(range(iterations), J_history)
plt.show()

# test hypothesis
temperature = 0.65
windspeed = 0.22
humidity = 0.6
x = np.matrix([1, temperature, windspeed, humidity])
h = x.dot(theta).item(0)

print('No. of probable bike rentals: ' + str(h))