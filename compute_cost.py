import numpy as np

def compute_cost(X, y, theta):
    # get size of training data
    m = y.shape[0]

    # get predictions
    h = X.dot(theta)

    # initialize J (cost)
    J = 0

    # iterate through predictions and compute squared error
    for i in range(0, m):
        print(str(i) + ' - ' + str(J))
        J += np.square(h.item(i) - y.item(i))

    return (1 / (2 * m)) * J
