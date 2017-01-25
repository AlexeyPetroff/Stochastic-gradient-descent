import numpy as np


def write_answer_to_file(answer, filename):
    with open(filename, 'w') as f_out:
        f_out.write(str(round(answer, 3)))


def mserror(y, y_pred):
    return np.sum((y - y_pred) ** 2) / (len(y))


def linear_prediction(X, w):
    return np.dot(X, w)


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    X_new = X[train_ind]
    answer = w - 2 * eta / X.shape[0] * X_new * (np.dot(w, X_new) - np.array(y)[train_ind])
    return answer


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    weight_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0
    np.random.seed(seed)

    while weight_dist > min_weight_dist and iter_num < max_iter:
        random_ind = np.random.randint(X.shape[0])
        temp_w = stochastic_gradient_step(X, y, w, random_ind, eta)
        err = mserror(np.array(y), linear_prediction(X, temp_w))
        errors.append(err)
        weight_dist = np.sqrt(np.sum((temp_w - w) ** 2))
        w = temp_w
        iter_num += 1
    return w, errors
