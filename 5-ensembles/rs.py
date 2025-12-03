import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from scipy import stats


class RandomSubspace(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator = GaussianNB(), n_estimators=10, n_features=2, voting="hard", random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.voting = voting
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)
        
    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        self.subspaces = self.random.choice(X.shape[1], 
                                        (self.n_estimators, self.n_features), replace=True)
        
        self.ensemble = [
            clone(self.base_estimator).fit(X[:, self.subspaces[i]], self.y_) 
            for i in range(self.n_estimators)
            ]
        
        # Return the classifier
        return self
    
    def predict_proba(self, X):
        if self.voting == "hard":
            self.probas_array = np.array([
                self.ensemble[i].predict(X[:, self.subspaces[i]]) 
                for i in range(self.n_estimators)
                ])
            self.preds = stats.mode(self.preds_array, axis=0)[0]
        else:
            self.probas_array = np.array([
                self.ensemble[i].predict(X[:, self.subspaces[i]]) 
                for i in range(self.n_estimators)
                ])
            self.mean_probas = np.mean(self.probas_array, axis=0)
            self.preds = np.argmax(self.mean_probas, axis=1)

        return self.preds
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)
        
        if self.voting  =="hard":
            self.preds_array = np.array([
                self.ensemble[i].predict(X[:, self.subspaces[i]]) 
                for i in range(self.n_estimators)
                ])
            self.preds = stats.mode(self.preds_array, axis=0)[0]
        else:
            pass
        
        return self.preds