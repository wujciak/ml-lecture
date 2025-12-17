import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import DistanceMetric


class KnoraU(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator = GaussianNB(), n_estimators=10, 
                voting="hard", k=7, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.voting = voting
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)
        self.k = k
        self.DistanceMetric = DistanceMetric.get_metric("euclidean")
        
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        self.estimators_ = []
        indxs = self.random.choice(X.shape[0], (self.estimators_, X.shape[0]), 
                                replace=True)
        
        for i in range(self.n_estimators):
            self.estimators_.append(
                clone(self.base_estimator).fit(self.X_[indxs[i]], self.y_[indxs[i]]), 
                replace=True)
        
        return self
    
    def competence(self, X, X_DSEL, y_DSEL):
        distance_matrix = self.dist.pairwise(X, X_DSEL)
        local_com_idxs =  np.argsort(distance_matrix, axis=1)[:, :self.k]
        
        # N_TEST x k_NEIGHBOURS x N_FEATURES
        X_local = X_DSEL[local_com_idxs]
        y_local = y_DSEL[local_com_idxs]
        
        estimator_weights = []
        for esitimator in self.estimators_:
            local_preds = np.array([
                esitimator.predict(X_local[i] for i in range(X_local.shape[0]))])
            
            estimator_weights.append(
                np.sum((local_preds[0] == y_local).astype(int), axis=1))
        
        return estimator_weights
    
    
    def predict(self, X, X_DSEL, y_DSEL):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)
        weights = self.competence(X, X_DSEL, y_DSEL)
        
        test_preds = np.array([estimator.predict[X] for estimator in self.estimators_])
        
        preds = []
        for i, test_sample_preds in test_preds.T:
            count = np.bincount(test_sample_preds, weights=weights.T[i])
            preds.append(np.argmax(count))
        
        return np.array(preds)
