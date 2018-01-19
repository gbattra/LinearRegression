def compute_cost(X, y, Theta):
    # get size of training data
    m = X.shape[0]

    # get predictions
    h = X * Theta

    # initialize J (cost)
    J = 0

    # iterate through predictions and compute squared error
    for i in range(0, m):
        J += (h.item(i) - y.item(i)) ** 2

    return (1 / (2 * m)) * J
