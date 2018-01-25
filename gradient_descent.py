from compute_cost import compute_cost
import numpy as np

def gradient_descent(X, y, theta, alpha, iterations):
    # length of training data
    m = X.shape[0]

    # track history of J
    J_history = np.zeros(iterations)

    # run gradient descent to find Theta's with lowest cost
    for iter in range(0, iterations):
        # compute predictions
        h = X.dot(theta)

        # initialize J
        J = 0

        # compute partial derivatives for each training example and add it to the cost
        for i in range(0, m):
            J = J + (h.item(i) - y.item(i)) * (X[i, :].T)

        theta = theta - alpha * ((1 / m) * J)
        J_history[iter] = compute_cost(X, y, theta)

    return [theta, J_history]
