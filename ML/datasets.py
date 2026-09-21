import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

# ─── Classification ────────────────────────────────────────────────────────────

def load_moons_clf(n=300, noise=0.2, seed=42):
    X, y = datasets.make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, y, ["x₁", "x₂"], ["Class A", "Class B"]

def load_circles_clf(n=300, noise=0.08, seed=42):
    X, y = datasets.make_circles(n_samples=n, noise=noise, random_state=seed, factor=0.5)
    return X, y, ["x₁", "x₂"], ["Inner", "Outer"]

def load_iris():
    d = datasets.load_iris()
    return d.data, d.target, list(d.feature_names), list(d.target_names)

def load_wine():
    d = datasets.load_wine()
    return d.data, d.target, list(d.feature_names), list(d.target_names)

def load_breast_cancer():
    d = datasets.load_breast_cancer()
    return d.data, d.target, list(d.feature_names), list(d.target_names)

def load_multilabel(seed=42):
    X, Y = datasets.make_multilabel_classification(
        n_samples=200, n_features=2, n_classes=3, n_labels=2,
        random_state=seed, allow_unlabeled=False)
    return X, Y, ["x₁", "x₂"], ["L1", "L2", "L3"]

# ─── Regression ────────────────────────────────────────────────────────────────

def load_linear_reg(n=120, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, (n, 1))
    y = 2.5 * X[:, 0] + 3 + rng.normal(0, 1.5, n)
    return X, y, ["x"], None

def load_nonlinear_reg(n=150, seed=42):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3 * np.pi, 3 * np.pi, (n, 1)), axis=0)
    y = np.sin(X[:, 0]) + 0.4 * np.sin(2 * X[:, 0]) + rng.normal(0, 0.3, n)
    return X, y, ["x"], None

def load_diabetes_reg():
    d = datasets.load_diabetes()
    return d.data, d.target, list(d.feature_names), None

# ─── Clustering ────────────────────────────────────────────────────────────────

def load_blobs(n=300, k=3, seed=42):
    X, y = datasets.make_blobs(n_samples=n, centers=k, random_state=seed, cluster_std=1.0)
    return X, y, ["x₁", "x₂"]

def load_moons_clust(n=300, noise=0.05, seed=42):
    X, y = datasets.make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, y, ["x₁", "x₂"]

def load_anisotropic(n=300, seed=42):
    X, y = datasets.make_blobs(n_samples=n, centers=3, random_state=seed)
    T = np.array([[0.6, -0.6], [-0.4, 0.8]])
    return X @ T, y, ["x₁", "x₂"]

def load_noisy(n=300, seed=42):
    X, y = datasets.make_blobs(n_samples=n, centers=4, cluster_std=2.0, random_state=seed)
    return X, y, ["x₁", "x₂"]

# ─── K-Modes synthetic categorical data ───────────────────────────────────────

def make_kmodes_data(seed=42):
    rng = np.random.default_rng(seed)
    centers = [(0, 0), (3, 3), (6, 0)]
    X = []
    for cx, cy in centers:
        pts = rng.integers([max(0, cx-1), max(0, cy-1)],
                           [cx+2, cy+2], size=(80, 2))
        X.append(pts)
    X = np.vstack(X)
    y = np.repeat([0, 1, 2], 80)
    return X, y, ["cat_A", "cat_B"]

# ─── CSV upload ────────────────────────────────────────────────────────────────

def process_csv(df, target_col, feature_cols, task, missing="drop"):
    df = df[feature_cols + [target_col]].copy()
    if missing == "Drop rows":
        df = df.dropna()
    else:
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
        df[target_col]   = df[target_col].fillna(df[target_col].mode().iloc[0])
    X = df[feature_cols].values.astype(float)
    if task in ("Classification", "Clustering"):
        le = LabelEncoder()
        y  = le.fit_transform(df[target_col].astype(str).values)
        class_names = list(le.classes_.astype(str))
    else:
        y = df[target_col].values.astype(float)
        class_names = None
    return X, y, feature_cols, class_names

# ─── Registries ────────────────────────────────────────────────────────────────

CLF_DATASETS   = ["Moons", "Circles", "Iris", "Wine", "Breast Cancer", "Upload CSV"]
REG_DATASETS   = ["Linear", "Non-linear", "Diabetes", "Upload CSV"]
CLUST_DATASETS = ["Blobs", "Moons", "Anisotropic", "Noisy Clusters", "Upload CSV"]

CLF_ALGOS  = [
    "KNN", "Naive Bayes", "Decision Tree", "Logistic Regression", "SVM",
    "Bagging", "Random Forest", "Gradient Boosting", "AdaBoost",
    "XGBoost", "LightGBM", "CatBoost (approx.)",
    "Voting Classifier", "Stacking", "Classifier Chain",
]
REG_ALGOS  = [
    "Linear Regression", "Ridge", "Lasso", "Elastic Net",
    "KNN Regressor", "Decision Tree Regressor", "Random Forest Regressor",
    "SVR", "Gradient Boosting Regressor",
]
CLUST_ALGOS = [
    "K-Means", "K-Medians", "K-Medoids", "K-Modes",
    "DBSCAN", "Hierarchical", "GMM",
]
STEP_ALGOS  = {"K-Means", "K-Medians", "K-Medoids", "K-Modes"}


def get_dataset(task, name, seed=42):
    """Return (X, y, feature_names, class_names). y=None for unsupervised."""
    if task == "Classification":
        return {
            "Moons":         lambda: load_moons_clf(seed=seed),
            "Circles":       lambda: load_circles_clf(seed=seed),
            "Iris":          load_iris,
            "Wine":          load_wine,
            "Breast Cancer": load_breast_cancer,
        }[name]()
    elif task == "Regression":
        r = {
            "Linear":      lambda: load_linear_reg(seed=seed),
            "Non-linear":  lambda: load_nonlinear_reg(seed=seed),
            "Diabetes":    load_diabetes_reg,
        }[name]()
        return r
    else:
        Xy_fn = {
            "Blobs":          lambda: load_blobs(seed=seed),
            "Moons":          lambda: load_moons_clust(seed=seed),
            "Anisotropic":    lambda: load_anisotropic(seed=seed),
            "Noisy Clusters": lambda: load_noisy(seed=seed),
        }[name]()
        X, y, feat = Xy_fn
        return X, y, feat, None
