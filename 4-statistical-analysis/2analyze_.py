# we need normal distribution for t-test
# critical difference diagram is useful for non-parametric tests like friedman test

import numpy as np
from scipy.stats import ttest_rel
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

clfs = {
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=1410)
}

scores = np.load("scores1_.npy")
print(scores)

alpha = .5

tstat = np.zeros((len(clfs), len(clfs)))
pvalue = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        tstat[i, j], pvalue[i, j] = ttest_rel(scores[i], scores[j])
        
print(tabulate(tstat, headers=clfs.keys(), tablefmt="latex_booktabs", floatfmt=".3f"))

advantage = np.zeros((len(clfs), len(clfs)))
advantage[tstat > 0] = 1
print(advantage)

significance = np.zeros((len(clfs), len(clfs)))
significance[pvalue <= alpha] = 1
print(significance)

stat_better = advantage * significance
print(np.mean(scores, axis=1))
print(stat_better)
