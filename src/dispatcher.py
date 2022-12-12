from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier

MODELS = {
    'Logistic_Regression': LogisticRegression(),
    "ExtraTressClassifer": ExtraTreeClassifier(n_estimator=200, n_jobs=-1, verbose=2),
    "RandomForestClassifer": RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2)
}