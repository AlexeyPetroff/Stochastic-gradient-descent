import pandas as pd
import numpy as np
import implementation


# We'll try to predict the sales in company, depending on the advertisment they use(TV, Radio, Newspaper)
adver_data = pd.read_csv('advertising.csv')
adver_data[['TV', 'Radio', 'Newspaper']]
X = adver_data[['TV', 'Radio', 'Newspaper']].values
y = adver_data.Sales

# Scale columns in matrix X
means, stds = X.mean(axis=0), X.std(axis=0)
X = (X - means) / stds
X = np.hstack((X, np.ones(len(X)).reshape(len(X), 1)))

stoch_grad_desc_weights, stoch_errors_by_iter = implementation.stochastic_gradient_descent(X, y, [0, 0, 0, 0],
                                                                                           max_iter=100000)
#New data
d = [151.5,	45.3,	53.5]
d = (d - means) / stds
d = np.append(d, [1.0])
answer = implementation.linear_prediction(d, stoch_grad_desc_weights)
print(answer)
