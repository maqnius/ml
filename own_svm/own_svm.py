import numpy as np
import pandas as pd


def test_own_svm():

    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.model_selection import train_test_split

    # create some data sets from
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html.

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.0, random_state=0),
                make_circles(noise=0.0, factor=0.5, random_state=1),
                linearly_separable
                ]

    # choose which dataset you want: '0' for moons, '1' for circles, '2' for linearly separable
    X_all = pd.DataFrame(datasets[0][0])
    y_all = pd.DataFrame(datasets[0][1])

    num_test = 0.2 # Part that is used for testing
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

    # Create test
    clf = own_svm(k = 0.001, C = 100.0)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    assert(type(predictions) == np.ndarray)

    assert(predictions.array_equal(y_test.as_matrix()))


class own_svm:

    def __init__(self, k, C):
        self.k = k
        self.C = C

    def fit(self, X_train, y_train):
        return None

    def predict(self, X):
        return None