from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np
import scipy.io
from pykernels import *
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

if False:
    dataset = fetch_california_housing()
    X = dataset.data
    X = X[1:15000,:]
    Y = dataset.target
    Y = Y[1:15000]
    Xtest = X[15001:20640,:]
    Ytest = Y[15001:20640]
    model = SVR(epsilon=0.1, kernel='rbf', gamma=5, C=1000)
    model.fit(X, Y)
    prediction = model.predict(Xtest)
    print("ok")

# Load data
mat = scipy.io.loadmat('santafe')
Z = mat['Z']

# Parameters
order = 23
C = 1000
sigma = 0.1
gamma = 1/sigma

# Windowing
# print(Z[0]) # array starts at 0 just to be sure you don't confuse with MATLAB
nb_X = len(Z) - order
X = np.zeros((nb_X,order,1))
Y = np.zeros((nb_X,1,1))
for i in range(len(Z)-order):
    X[i,:] = Z[i:i+order]
    Y[i] = Z[i+order]
X = np.reshape(X, (nb_X, order))
Y = np.reshape(Y, (nb_X))

# Train model
kernel_type = 'rbf'
if kernel_type == 'rbf':
    param_grid = {"C": np.linspace(10 ** (-2), 10 ** 3, 4), 'gamma': np.linspace(0.0001, 1, 5)}
    mod = SVR(epsilon=0.1, kernel='rbf')
    model = GridSearchCV(estimator=mod, param_grid=param_grid, scoring="neg_mean_squared_error", verbose=0)
elif kernel_type == 'anova':
    param_grid = {"C": np.linspace(10 ** (-2), 10 ** 3, 4), 'gamma': np.linspace(0.01, 10, 5)} #'d': np.asarray([2,3,4])
    model = SVR(epsilon=0.1, kernel=ANOVA(sigma=14.0, d=5), gamma=8000)
    model = GridSearchCV(estimator=model, param_grid=param_grid, scoring="neg_mean_squared_error", verbose=0)
model.fit(X,Y)
print(len(model.best_estimator_.support_vectors_))

# Visualise
plt.plot(X, 'b.')
plt.plot(model.predict(X), 'r-')
plt.show()

# Predict
test_set = mat['Ztest']
nb_predictions = len(test_set)
predictions = np.zeros(nb_predictions)
current = Z[len(Z)-order:len(Z)]
current = np.reshape(current, (1,-1))
current = current.tolist()
for i in range(nb_predictions):
    prediction = model.predict(current)
    predictions[i] = prediction
    current = [current[0][1:order] + prediction.tolist()]

# Visualize
plt.plot(test_set, 'b-')
plt.plot(predictions, 'r-')
plt.show()