import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from compute_cost import compute_cost

# this is a basic linear regression model
# it predicts the length of the bike trip by the age of the rider
data = pd.read_csv('dataset.csv').values
X = np.matrix(data[:, 0:1])
y = np.matrix(data[:, 1])

# preprocess X (convert it from year to age)
for i in range(0, X.size):
    X[i] = datetime.datetime.now().year - X[i]

# plot the data
plt.plot(X, y, 'ro')

# add ones
X0 = np.ones((X.size, 1))
X = np.hstack(X0, X)

# initialize thetas
Theta = np.zeros((X.shape[1], 1))

# initialize learning parameters
iterations = 1500
alpha = 0.1

# testing cost function
J = compute_cost(X, y, Theta)