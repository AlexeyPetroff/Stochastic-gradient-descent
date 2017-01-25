import pandas as pd
import numpy as np


def write_answer_to_file(answer, filename):
    with open(filename, 'w') as f_out:
        f_out.write(str(round(answer, 3)))


adver_data = pd.read_csv('advertising.csv')

adver_data[['TV', 'Radio', 'Newspaper']]

X = adver_data[['TV', 'Radio', 'Newspaper']].values
y = adver_data.Sales

means, stds = X.mean(axis=0), X.std(axis=0)

X = (X - means) / stds

X = np.hstack((X, np.ones(len(X)).reshape(len(X), 1)))


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
    return w, errors, iter_num


stoch_grad_desc_weights, stoch_errors_by_iter, test = stochastic_gradient_descent(X, y, [0, 0, 0, 0], max_iter=100000)
print(stoch_errors_by_iter)
print(test)
