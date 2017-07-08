import numpy as np
import pandas as pd


def test_own_svm():
    from sklearn.datasets import make_moons, make_circles, make_classification

    clf = own_svm()

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
    X_test = pd.DataFrame(datasets[0][0])
    y_test = pd.DataFrame(datasets[0][1])





class own_svm:
    def fit(self):
        return None