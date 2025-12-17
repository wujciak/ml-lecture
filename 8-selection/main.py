from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from knorau import KnoraU

X, y = make_classification(n_samples=1000, n_features=10,
                        n_informative=10, n_redundant=0, n_repeated=0,
                        random_state=1410, class_sep=.4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42)

X_train, X_DSEL, y_train, y_DSEL = train_test_split(
    X_train, y_train, test_size=.2, random_state=42)

# print(X_train.shape, X_DSEL.shape, X_test.shape)

clf = KnoraU(n_estimators=10, voting="hard", random_state=42)
preds = clf.predict(X_test, X_DSEL, y_DSEL)
print(balanced_accuracy_score(y_test, preds))
