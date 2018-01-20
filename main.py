import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import scipy
import datacleaner
from compute_cost import compute_cost
from gradient_descent import gradient_descent

# this is a basic linear regression model
# it predicts the length of the bike trip by the age of the rider
data = pd.read_csv('dataset_linear.csv')
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:1])
y = np.matrix(clean_data[:, 1:2])

# preprocess X (convert it from year to age)
for i in range(0, X.size):
    X[i] = datetime.datetime.now().year - X[i]


# plot the data
plt.plot(X, y, 'ro')

# add ones
X0 = np.ones((X.size, 1))
# X2 = np.square(X)
X = np.hstack((X0, X))

# initialize thetas
theta = np.zeros((X.shape[1], 1))

# initialize learning parameters
iterations = 100
alpha = 0.000000001

# test cost function
J = compute_cost(X, y, theta)

# train Thetas to fit data
[theta, J_history] = gradient_descent(X, y, theta, alpha, iterations)

h = X.dot(theta)

