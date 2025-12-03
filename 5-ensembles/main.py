from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from rs import RandomSubspace

X, y = make_classification(n_samples=1000, n_features=10,
                        n_informative=10, n_redundant=0, n_repeated=0,
                        random_state=1410,class_sep=.4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42
    )

clf = RandomSubspace().fit(X_train, y_train)
score = accuracy_score(y_test, clf.predict(X_test))
print(score)
