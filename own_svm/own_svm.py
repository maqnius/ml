import numpy as np
import pandas as pd
from random import randint

class own_smo:

    def __init__(self, k, C):
        self.k = k
        self.C = C

    def fit(self, X_train, y_train):
        return None

    def predict(self, X):
        return None

class own_smo_simple:

    def __init__(self, C):
        self.C = C
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.n_test_samples = 0
        self.b = 0.

    def fit(self, X_train, y_train, max_passes=10, tol=1e-4):
        # Convert arguments to numpy arrays if they are in pandas datastructures
        if type(X_train) == pd.DataFrame:
            self.X_train = X_train.as_matrix()
        if type(y_train) == pd.DataFrame or type(y_train) == pd.Series:
            self.y_train = y_train.as_matrix().squeeze(axis = 1)

        # Create array for storing the alpha values
        self.n_test_samples = len(y_train)
        self.alpha = np.zeros(self.n_test_samples)

        passes = 0 # Counting runs without changing a value
        while passes < max_passes:
            changed_alpha = False

            for i in range(self.n_test_samples):
                E_i = self.dec_func(i) - self.y_train[i]

                if (self.y_train[i] * E_i < -tol and self.alpha[i] < self.C) or \
                        (self.y_train[i] * E_i > tol and self.alpha[i] > 0):

                    j = randint(0, self.n_test_samples - 1)

                    # Saving old alphas
                    a_i_old = self.alpha[i]
                    a_j_old = self.alpha[j]

                    E_j = self.dec_func(j) - self.y_train[j]
                    L, H = self.calc_bonds(i, j)

                    if L == H:
                        continue

                    eta = 2 * self.kernel_ind(i, j) - self.kernel_ind(i, i) - self.kernel_ind(j, j)
                    if eta >= 0:
                        continue

                    a_j = a_j_old - self.y_train[j] * (E_i - E_j) / eta

                    if a_j > H:
                        a_j = H
                    elif a_j < L:
                        a_j = L

                    if np.abs(a_j - a_j_old) < tol:  # Lets check it
                        continue

                    a_i = a_i_old + self.y_train[i] * self.y_train[j] * (a_j_old - a_j)

                    # Calculate new threshold
                    d_a_i = self.y_train[i] * (a_i - a_i_old)
                    d_a_j = self.y_train[j] * (a_j - a_j_old)
                    b_1 = self.b - E_i - d_a_i * self.kernel_ind(i, i) - d_a_j * self.kernel_ind(i, j)
                    b_2 = self.b - E_j - d_a_i * self.kernel_ind(i, j) - d_a_j * self.kernel_ind(j, j)

                    if self.C > a_i > 0.:
                        self.b = b_1
                    elif self.C > a_j > 0:
                        self.b = b_2
                    else:
                        self.b = (b_1 + b_2)/2

                    self.alpha[i] = a_i
                    self.alpha[j] = a_j
                    changed_alpha = True

            passes += 1 if changed_alpha else 0

    def calc_bonds(self, i, j):
        if self.y_train[i] != self.y_train[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

        return L, H


    def dec_func(self, x_ind):
        sum = - self.b
        for i in range(self.n_test_samples):
            sum += self.alpha[i] * self.y_train[i] * self.kernel_ind(i, x_ind)

        return sum

    def kernel_ind(self, i, j):
        return self.kernel(self.X_train[i], self.X_train[j])

    def kernel(self, x, y):
        return x.dot(y)

    def predict(self, X):
        # Convert to numpy arrays
        if type(X) == pd.DataFrame or type(X) == pd.Series:
            X = X.as_matrix()

        y = np.zeros(X.shape[0])
        for i in range(len(y)):
            y[i] = self.b
            for j in range(self.n_test_samples):
                y[i] += self.alpha[j]*self.y_train[j]*self.kernel(self.X_train[j], X[i])
        return np.sign(y).astype(int)