import numpy as np
import pandas as pd
from random import randint


def svm_test(smo):

    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.model_selection import train_test_split

    # create some data sets from
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html.

    X, y = make_classification(n_samples=20, n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.0, random_state=0),
                make_circles(noise=0.0, factor=0.5, random_state=1),
                linearly_separable
                ]

    # choose which dataset you want: '0' for moons, '1' for circles, '2' for linearly separable
    X_all = pd.DataFrame(datasets[2][0])
    y_all = pd.DataFrame(datasets[2][1])

    num_test = 0.2 # Part that is used for testing
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

    # Create test
    clf = smo(C = 100.0)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    assert(type(predictions) == np.ndarray)

    assert(np.array_equal(predictions, y_test.as_matrix()))

#def test_own_smo():
#    svm_test(own_smo)

def test_own_smo_simple():
    svm_test(own_smo_simple)

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
        self.b = 0.

    def fit(self, X_train, y_train, max_passes=10, tol=1e-3):
        # Convert arguments to numpy arrays if they are in pandas datastructures
        if type(X_train) == pd.DataFrame:
            self.X_train = X_train.as_matrix()
        if type(y_train) == pd.DataFrame or type(y_train) == pd.Series:
            self.y_train = y_train.as_matrix()

        # Create array for storing the alpha values
        self.alpha = np.zeros(len(y_train))

        passes = 0 # Counting runs without changing a value
        while passes < max_passes:
            changed_alpha = False

            for i in range(len(self.y_train)):
                E_i = self.dec_func(i) - self.y_train[i]
                if (self.y_train[i] * E_i < -tol and self.alpha[i] < self.C) or \
                        (self.y_train[i] * E_i > tol and self.alpha[i] > 0):

                    j = randint(0, len(y_train) - 1)

                    # Saving old alphas
                    a_i_old = self.alpha[i]
                    a_j_old = self.alpha[j]

                    E_j = self.dec_func(j) - self.y_train[j]
                    L = self.calc_L(i, j)
                    H = self.calc_H(i, j)

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

                    changed_alpha = True

            passes += 1 if changed_alpha else 0

    def calc_L(self, i, j):
        if self.y_train[i] != self.y_train[j]:
            return max(0, self.alpha[j] - self.alpha[i])
        else:
            return max(0, self.alpha[i] + self.alpha[j] - self.C)

    def calc_H(self, i, j):
        if self.y_train[i] != self.y_train[j]:
            return min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            return min(self.C, self.alpha[i] + self.alpha[j])


    def dec_func(self, x_ind):
        sum = self.b
        for i in range(len(self.y_train)):
            sum += self.alpha[i] * self.y_train[i] * self.kernel_ind(i, x_ind)

        return sum

    def kernel_ind(self, i, j):
        return self.kernel(self.X_train[i], self.X_train[j])

    def kernel(self, x, y):
        return x.dot(y)

    def predict(self, X):
        X = X.as_matrix()
        y = np.zeros(X.shape[0])
        for i in range(len(y)):
            y[i] = self.b
            for j in range(len(self.y_train)):
                y[i] += self.alpha[j]*self.y_train[j]*self.kernel(self.X_train[j], X[i])
        return y