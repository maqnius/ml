#   Copyright (C) 2017 Mark Niehues, Stefaan Hessmann
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#


import numpy as np
import pandas as pd
from random import randint
from kernels import Kernels

class own_smo:
    """
    Going to be a more advanced method of choosing the alphas
    """
    def __init__(self, C):
        self.C = C

    def fit(self, X_train, y_train):
        return None

    def predict(self, X):
        return None

class own_smo_simple:
    """
    Implementation of a simple smo algorithm that chooses the comparable pairs randomly

    Main Source: http://cs229.stanford.edu/materials/smo.pdf
    """
    def __init__(self, C = 50.0, gamma = None):

        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.n_test_samples = 0
        self.b = 0.
        self.C = C

        self.min_label = -1

        self.kernel_set = Kernels(gamma)
        self.kernel = None

    def fit(self, X_train, y_train, max_passes=10, tol=1e-4, kernel="rbf"):
        """
        Fits alpha values and the threshold b with given Training data

        Parameters
        ----------
        X_train: Numpy Array or Pandas Data Set
            Training Data Set
        y_train: Numpy Array oder Pandas Series
            Labels for Training
        max_passes: int
            Maximal Number of runs without any change in the alpha values that
            determines the end of fitting
        tol: float
            Tolerance on estimated Error
        """



        # Convert arguments to numpy arrays if they are in pandas datastructures
        if type(X_train) == pd.DataFrame:
            self.X_train = X_train.as_matrix()
        if type(y_train) == pd.DataFrame or type(y_train) == pd.Series:
            self.y_train = y_train.as_matrix().squeeze(axis = 1)

        self.n_test_samples = len(y_train)

        # Set Kernel
        self.kernel = self.kernel_set.get_kernel(kernel)
        if kernel == "rbf" and self.kernel_set.gamma is None:
            self.kernel_set.gamma = 1 / self.n_test_samples

        # QUICK AND DIRTY
        # Detect if the lables are [1, 0] instead of [1, -1] and correct them
        if min(self.y_train) == 0:
            self.min_label = 0
            self.y_train = self.y_train * 2 - 1

        # Create array for storing the alpha values
        self.alpha = np.zeros(self.n_test_samples)

        passes = 0  # Counting runs without changing a value
        while passes < max_passes:
            changed_alpha = False

            for i in range(self.n_test_samples):
                # Calculate the error with the current alpha
                E_i = self.dec_func(self.X_train[i]) - self.y_train[i]

                # If accuracy is not satisfying yet
                if (self.y_train[i] * E_i < -tol and self.alpha[i] < self.C) or \
                        (self.y_train[i] * E_i > tol and self.alpha[i] > 0):

                    # Randomly choose another alpha to pair
                    j = randint(0, self.n_test_samples - 1)

                    # Saving old alphas
                    a_i_old = self.alpha[i]
                    a_j_old = self.alpha[j]

                    # Calculate the error of the other alpha
                    E_j = self.dec_func(self.X_train[j]) - self.y_train[j]

                    # Calculate the valid limits that are a consequence of the linear dependence
                    L, H = self.calc_limits(i, j)
                    if L == H:
                        continue

                    # Evaluate the second derivative of the Loss function for optimizing
                    # Eta should be negative to make shore, what we are approaching a maximum
                    eta = 2 * self.kernel_ind(i, j) - self.kernel_ind(i, i) - self.kernel_ind(j, j)
                    if eta >= 0:
                        continue

                    a_j = a_j_old - self.y_train[j] * (E_i - E_j) / eta

                    # Clip the new alpha to the limits
                    if a_j > H:
                        a_j = H
                    elif a_j < L:
                        a_j = L

                    if np.abs(a_j - a_j_old) < tol:
                        continue

                    # Calculate the new value for a_i from the new value of a_j
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

                    # Replace the alpha values
                    self.alpha[i] = a_i
                    self.alpha[j] = a_j
                    changed_alpha = True

            passes += 1 if changed_alpha else 0

    def calc_limits(self, i, j):
        """
        Calculates the limits for the new alpha value that follow from
        the linear dependence of two terms

        """
        if self.y_train[i] != self.y_train[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

        return L, H

    def dec_func(self, x):
        """ Decision function without the signum
        """
        sum = self.b
        for i in range(self.n_test_samples):
            sum += self.alpha[i] * self.y_train[i] * self.kernel(self.X_train[i], x)

        return sum

    def predict(self, X):
        """
        Predicts the Class for an X
        Parameters
        ----------
        X: Array or Float
            Points whose class shall be predicted

        Returns
        -------
        Integer Array or Integer
            Predicted classes [1, -1]
        """
        # Convert to numpy arrays
        if type(X) == pd.DataFrame or type(X) == pd.Series:
            X = X.as_matrix()

        # Initialize
        y = np.zeros(X.shape[0])
        for i in range(len(y)):
            # Apply the decision function to each point
            y[i] = self.dec_func(X[i])

        y = np.sign(y).astype(int)

        # If given Train format was labeled with [0,1]
        # return the labeles in the same way
        if self.min_label == 0:
            y = (y + 1) // 2

        return y

    def kernel_ind(self, i, j):
        """
        Wrapper that makes it possible to call the kernel only with the indices
        of the train set.
        """
        return self.kernel(self.X_train[i], self.X_train[j])