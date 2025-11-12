import numpy as np
from sklearn import clone
from  sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

clfs = {
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=1410)
}

data = np.genfromtxt('datasets/Australian.csv', delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)
print(X, y)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1410)

# CLFS, FOLDS
scores = np.zeros((len(clfs), 10)) 
for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    for clf_id, clf_name in enumerate(clfs.keys()):
        clf = clone(clfs[clf_name]).fit(X_train, y_train)
        preds = clf.predict(X_test)
        score = accuracy_score(y_test, preds)
        scores[clf_id, i] = score

print(scores)
np.save("scores1_", scores)
