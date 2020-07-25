# Adapted from: https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np

# generate regression dataset
true_c = 5
X, Y, coef = make_regression(n_samples=100, n_features=1, bias=true_c, noise=0.5, coef=True)
X = X.reshape(100,) # Make X one-dimensional, intead of two-dimensional with 1 column

print(f"True m: {coef}")
print(f"True c: {true_c}")

# Building the model
m = 0
c = 0

L = 0.01 # The learning rate
epochs = 1000 # The number of iterations to perform

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent
for i in range(epochs):
    Y_pred = m*X + c # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred)) # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred) # Derivative wrt c
    m = m - L * D_m # Update m
    c = c - L * D_c # Update c

print(f"Estimated m: {m}")
print(f"Estimated c: {c}")

# Making predictions
Y_pred = m*X+c

plt.scatter(X,Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()
