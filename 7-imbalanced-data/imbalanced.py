from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

X, y = make_classification(
    n_samples=1000, n_features=50, n_informative=50, n_redundant=0, n_repeated=0, random_state=None, class_sep=3, weights=[0.9,0.1]
    )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

fig, ax = plt.subplots(2, 2, figsize=(13, 13))
ax = ax.ravel()

ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=50)
ax[0].set_title("Original Data Distribution", fontsize=14)
ax[0].grid(ls=":", c=(.7, .7, .7))

# imbalanced-learn library

# Undersampling - RandomUnderSampling
over = RandomUnderSampler(sampling_strategy="auto", random_state=42)
X_under, y_under = over.fit_resample(X, y)
ax[1].scatter(X_under[:, 0], X_under[:, 1], c=y_under, cmap="coolwarm", s=50)
ax[1].set_title("Undersampled Data Distribution", fontsize=14)
ax[1].grid(ls=":", c=(.7, .7, .7))



# Oversampling - RandomOverSampling or SMOTE
over = RandomOverSampler(sampling_strategy="auto", random_state=42)
X_over, y_over = over.fit_resample(X, y)
ax[2].scatter(X_over[:, 0], X_over[:, 1], c=y_over, cmap="coolwarm", s=50)
ax[2].set_title("Oversampled Data Distribution", fontsize=14)
ax[2].grid(ls=":", c=(.7, .7, .7))


over = SMOTE(sampling_strategy="auto", random_state=42)
X_overS, y_overS = over.fit_resample(X, y)
ax[3].scatter(X_overS[:, 0], X_overS[:, 1], c=y_overS, cmap="coolwarm", s=50)
ax[3].set_title("Oversampled Data Distribution - SMOTE", fontsize=14)
ax[3].grid(ls=":", c=(.7, .7, .7))

plt.tight_layout()
plt.show()
plt.close()
