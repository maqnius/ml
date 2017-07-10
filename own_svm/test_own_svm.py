import numpy as np
import pandas as pd
from own_svm import own_smo_simple, own_smo


def svm_test(smo):
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.model_selection import train_test_split

    # create some data sets from
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html.

    X, y = make_classification(n_samples=20, n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    y = y*2-1  # Change lables from [1,0] to [1, -1]

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
    clf = smo(C = 1.0)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    assert(type(predictions) == np.ndarray)

    assert(np.array_equal(predictions, y_test.as_matrix()))


#def test_own_smo():
#    svm_test(own_smo)


def test_own_smo_simple():
    svm_test(own_smo_simple)
