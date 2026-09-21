"""Model factories and step-based algorithm implementations."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.multioutput import ClassifierChain
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# ─── Step-based clustering (manual implementations) ───────────────────────────

def _init_centers(X, k, method, seed):
    rng = np.random.default_rng(seed)
    if "k-means++" in method:
        c = [X[rng.choice(len(X))]]
        for _ in range(k - 1):
            d = np.min([np.sum((X - ci) ** 2, axis=1) for ci in c], axis=0)
            c.append(X[rng.choice(len(X), p=d / d.sum())])
        return np.array(c)
    return X[rng.choice(len(X), k, replace=False)].copy()


def kmeans_steps(X, k, init="k-means++", max_iter=50, seed=42):
    centers = _init_centers(X, k, init, seed)
    steps = []
    for _ in range(max_iter):
        dists  = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        steps.append((centers.copy(), labels.copy()))
        new_c = np.array([X[labels == i].mean(0) if (labels == i).any() else centers[i]
                          for i in range(k)])
        if np.allclose(centers, new_c, atol=1e-8):
            steps.append((new_c.copy(), labels.copy()))
            break
        centers = new_c
    return steps


def kmedians_steps(X, k, max_iter=50, seed=42):
    rng     = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    steps   = []
    for _ in range(max_iter):
        dists  = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        steps.append((centers.copy(), labels.copy()))
        new_c = np.array([np.median(X[labels == i], axis=0) if (labels == i).any() else centers[i]
                          for i in range(k)])
        if np.allclose(centers, new_c, atol=1e-8):
            steps.append((new_c.copy(), labels.copy()))
            break
        centers = new_c
    return steps


def kmedoids_steps(X, k, max_iter=50, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    steps = []
    for _ in range(max_iter):
        dists  = np.linalg.norm(X[:, None] - X[idx][None], axis=2)
        labels = np.argmin(dists, axis=1)
        steps.append((X[idx].copy(), labels.copy()))
        new_idx = idx.copy()
        for i in range(k):
            cl = np.where(labels == i)[0]
            if len(cl) == 0:
                continue
            D = np.sum(np.linalg.norm(X[cl][:, None] - X[cl][None], axis=2), axis=1)
            new_idx[i] = cl[np.argmin(D)]
        if np.all(new_idx == idx):
            break
        idx = new_idx
    return steps


def kmodes_steps(X_int, k, max_iter=50, seed=42):
    rng   = np.random.default_rng(seed)
    modes = X_int[rng.choice(len(X_int), k, replace=False)].copy().astype(float)
    steps = []
    for _ in range(max_iter):
        dists  = np.array([[np.sum(x != m) for m in modes] for x in X_int])
        labels = np.argmin(dists, axis=1)
        steps.append((modes.copy(), labels.copy()))
        new_m = np.array([
            np.apply_along_axis(lambda c: np.bincount(c.astype(int)).argmax(), 0,
                                X_int[labels == i])
            if (labels == i).any() else modes[i]
            for i in range(k)
        ]).astype(float)
        if np.all(new_m == modes):
            break
        modes = new_m
    return steps

# ─── Classifier factory ────────────────────────────────────────────────────────

def make_classifier(name, p, seed=42):
    dt3 = lambda: DecisionTreeClassifier(max_depth=3, random_state=seed)
    lr  = lambda: LogisticRegression(max_iter=1000, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=p["K"], metric=p["metric"], weights=p["weights"])
    if name == "Naive Bayes":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if name == "Decision Tree":
        return DecisionTreeClassifier(criterion=p["criterion"].lower(),
                                      max_depth=p["max_depth"] or None,
                                      min_samples_split=p["min_samples_split"],
                                      random_state=seed)
    if name == "Logistic Regression":
        return LogisticRegression(C=p["C"], penalty=p["penalty"],
                                  max_iter=p["max_iter"], random_state=seed,
                                  solver="saga" if p["penalty"] == "l1" else "lbfgs")
    if name == "SVM":
        g = p["gamma"]
        gamma = float(g) if g not in ("scale", "auto") else g
        return SVC(kernel=p["kernel"], C=p["C"], gamma=gamma,
                   probability=True, random_state=seed)
    if name == "Bagging":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=p["base_depth"], random_state=seed),
            n_estimators=p["n_estimators"], max_samples=p["sample_fraction"], random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"],
                                      max_depth=p["max_depth"] or None,
                                      max_features=p["max_features"], random_state=seed)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=p["n_estimators"],
                                          learning_rate=p["lr"],
                                          max_depth=p["max_depth"], random_state=seed)
    if name == "AdaBoost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=p["base_depth"], random_state=seed),
            n_estimators=p["n_estimators"], learning_rate=p["lr"], random_state=seed)
    if name == "XGBoost":
        if _HAS_XGB:
            return XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["lr"],
                                 max_depth=p["max_depth"], random_state=seed,
                                 eval_metric="logloss", verbosity=0)
        return GradientBoostingClassifier(n_estimators=p["n_estimators"],
                                          learning_rate=p["lr"], max_depth=p["max_depth"],
                                          random_state=seed)
    if name == "LightGBM":
        if _HAS_LGB:
            return LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=p["lr"],
                                  num_leaves=p["num_leaves"], random_state=seed, verbosity=-1)
        return GradientBoostingClassifier(n_estimators=p["n_estimators"],
                                          learning_rate=p["lr"], max_depth=4, random_state=seed)
    if name == "CatBoost (approx.)":
        return GradientBoostingClassifier(n_estimators=p["iterations"],
                                          learning_rate=p["lr"], max_depth=p["depth"],
                                          random_state=seed)
    if name == "Voting Classifier":
        return VotingClassifier(
            estimators=[("knn", KNeighborsClassifier(5)),
                        ("dt",  dt3()),
                        ("lr",  lr())],
            voting=p["voting"])
    if name == "Stacking":
        return StackingClassifier(
            estimators=[("knn", KNeighborsClassifier(5)),
                        ("dt",  dt3()),
                        ("svm", SVC(probability=True, random_state=seed))],
            final_estimator=lr(), cv=p["cv_folds"])
    if name == "Classifier Chain":
        return ClassifierChain(dt3(), order=p["order"], random_state=seed)
    raise ValueError(name)

# ─── Regressor factory ─────────────────────────────────────────────────────────

def make_regressor(name, p, seed=42):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=p["alpha"])
    if name == "Lasso":
        return Lasso(alpha=p["alpha"], max_iter=5000)
    if name == "Elastic Net":
        return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=5000)
    if name == "KNN Regressor":
        return KNeighborsRegressor(n_neighbors=p["K"], metric=p["metric"], weights=p["weights"])
    if name == "Decision Tree Regressor":
        return DecisionTreeRegressor(max_depth=p["max_depth"] or None,
                                     min_samples_split=p["min_samples_split"],
                                     random_state=seed)
    if name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=p["n_estimators"],
                                     max_depth=p["max_depth"] or None,
                                     max_features=p["max_features"], random_state=seed)
    if name == "SVR":
        g = p["gamma"]
        gamma = float(g) if g not in ("scale", "auto") else g
        return SVR(kernel=p["kernel"], C=p["C"], gamma=gamma)
    if name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(n_estimators=p["n_estimators"],
                                         learning_rate=p["lr"], max_depth=p["max_depth"],
                                         random_state=seed)
    raise ValueError(name)

# ─── Cluster model factory ─────────────────────────────────────────────────────

def make_cluster_model(name, p, seed=42):
    if name == "DBSCAN":
        return DBSCAN(eps=p["eps"], min_samples=p["min_samples"], metric=p["metric"])
    if name == "Hierarchical":
        return AgglomerativeClustering(n_clusters=p["n_clusters"], linkage=p["linkage"])
    if name == "GMM":
        return GaussianMixture(n_components=p["n_components"],
                               covariance_type=p["cov_type"],
                               max_iter=200, random_state=seed)
    return None

# ─── Algorithm descriptions ────────────────────────────────────────────────────

DESCRIPTIONS = {
    "KNN": {
        "how": "Classifies a point by majority vote among its K nearest neighbors in the training set.",
        "params": {
            "K": "Number of neighbors. Small K → complex boundary; large K → smoother boundary.",
            "Distance metric": "How distance is measured. Euclidean is standard; Manhattan is less sensitive to outliers.",
            "Voting": "Uniform weights all neighbors equally. Distance weights closer neighbors more.",
        }
    },
    "Naive Bayes": {
        "how": "Assumes features are conditionally independent given the class, then applies Bayes' theorem to compute class probabilities.",
        "params": {
            "Var smoothing": "Adds a small fraction of the largest variance to all variances, preventing zero-probability issues.",
        }
    },
    "Decision Tree": {
        "how": "Recursively splits the feature space with axis-aligned cuts that best separate classes, forming a tree of if-then rules.",
        "params": {
            "Criterion": "Gini measures impurity; Entropy uses information gain. Results are usually similar.",
            "Max depth": "Limits tree depth. Shallow trees underfit; deep trees overfit.",
            "Min samples split": "Minimum samples needed to split a node. Higher values prevent overfitting.",
        }
    },
    "Logistic Regression": {
        "how": "Fits a linear decision boundary by learning weights that maximize the log-likelihood of the training labels.",
        "params": {
            "C": "Inverse regularization strength. Large C → fits training data closely; small C → smoother boundary.",
            "Penalty": "L2 shrinks all weights; L1 drives some weights to exactly zero (feature selection).",
            "Max iterations": "Training convergence budget.",
        }
    },
    "SVM": {
        "how": "Finds the hyperplane that maximizes the margin between classes. The kernel trick maps data to higher dimensions for non-linear boundaries.",
        "params": {
            "Kernel": "Linear: straight boundary. RBF: radial basis → flexible. Poly: polynomial surface.",
            "C": "Penalty for misclassification. High C → less margin, fewer errors; low C → larger margin, more tolerance.",
            "Gamma": "RBF/Poly bandwidth. High gamma → local influence; low gamma → global influence.",
        }
    },
    "Bagging": {
        "how": "Trains multiple base learners on random bootstrap samples and aggregates their predictions by majority vote.",
        "params": {
            "Estimators": "Number of base models. More estimators → more stable but slower.",
            "Sample fraction": "Fraction of data used per bootstrap sample.",
            "Base depth": "Complexity of each base decision tree.",
        }
    },
    "Random Forest": {
        "how": "An ensemble of decision trees, each trained on a bootstrap sample with a random subset of features at each split, reducing correlation between trees.",
        "params": {
            "Trees": "More trees → more stable predictions but diminishing returns.",
            "Max depth": "Controls individual tree complexity.",
            "Max features": "'sqrt' uses √p features per split; 'log2' uses log₂p.",
        }
    },
    "Gradient Boosting": {
        "how": "Builds trees sequentially, each correcting the residual errors of the previous ensemble.",
        "params": {
            "Estimators": "Number of boosting rounds.",
            "Learning rate": "Shrinks each tree's contribution. Smaller → more trees needed, often better.",
            "Max depth": "Depth of each weak learner. Shallow trees (3-5) work best.",
        }
    },
    "AdaBoost": {
        "how": "Trains weak learners sequentially, increasing the weight of misclassified samples so each learner focuses on hard examples.",
        "params": {
            "Estimators": "Number of weak learners.",
            "Learning rate": "Shrinks contribution of each learner.",
            "Base depth": "Depth of base stumps (1 = decision stumps).",
        }
    },
    "XGBoost": {
        "how": "Optimized gradient boosting with regularization, parallel tree construction, and hardware acceleration.",
        "params": {
            "Estimators": "Number of boosting rounds.",
            "Learning rate": "Step size shrinkage.",
            "Max depth": "Maximum tree depth.",
        }
    },
    "LightGBM": {
        "how": "Gradient boosting that grows trees leaf-wise (best-first) rather than level-wise, making it faster on large datasets.",
        "params": {
            "Estimators": "Number of boosting rounds.",
            "Learning rate": "Step size shrinkage.",
            "Num leaves": "Maximum number of leaves per tree (controls complexity).",
        }
    },
    "CatBoost (approx.)": {
        "how": "Gradient boosting optimized for categorical features with ordered boosting (approximated here with sklearn GradientBoosting).",
        "params": {
            "Iterations": "Number of trees.",
            "Learning rate": "Step size shrinkage.",
            "Depth": "Maximum tree depth.",
        }
    },
    "Voting Classifier": {
        "how": "Combines predictions of KNN + Decision Tree + Logistic Regression by majority vote (hard) or averaged probabilities (soft).",
        "params": {
            "Voting": "Hard: majority class label. Soft: average predicted probabilities (usually better).",
        }
    },
    "Stacking": {
        "how": "Base models (KNN, DT, SVM) generate predictions that are fed as features to a meta-learner (Logistic Regression).",
        "params": {
            "CV folds": "Cross-validation folds for generating out-of-fold predictions for the meta-learner.",
        }
    },
    "Classifier Chain": {
        "how": "For multilabel problems: trains a chain of classifiers where each uses previous label predictions as additional input features.",
        "params": {
            "Chain order": "Random: random label order. Classifier predicts label 1, uses it for label 2, and so on.",
        }
    },
    "Linear Regression": {
        "how": "Finds the line (or hyperplane) that minimizes the sum of squared residuals between predictions and true values.",
        "params": {}
    },
    "Ridge": {
        "how": "Linear regression with L2 regularization that penalizes large coefficients, preventing overfitting.",
        "params": {
            "Alpha": "Regularization strength. Higher → more shrinkage of coefficients.",
        }
    },
    "Lasso": {
        "how": "Linear regression with L1 regularization that drives some coefficients to exactly zero, performing feature selection.",
        "params": {
            "Alpha": "Regularization strength. Higher → more coefficients become zero.",
        }
    },
    "Elastic Net": {
        "how": "Combines L1 and L2 regularization (Lasso + Ridge), useful when there are many correlated features.",
        "params": {
            "Alpha": "Overall regularization strength.",
            "L1 ratio": "Balance between L1 and L2. 0 = Ridge, 1 = Lasso.",
        }
    },
    "KNN Regressor": {
        "how": "Predicts a value by averaging the K nearest neighbors' target values.",
        "params": {
            "K": "Number of neighbors. Small K → wiggly predictions; large K → smoother.",
            "Distance metric": "How neighbor distance is measured.",
            "Voting": "Uniform: simple average. Distance: closer neighbors weighted more.",
        }
    },
    "Decision Tree Regressor": {
        "how": "Recursively partitions the input space and predicts the mean target value within each leaf.",
        "params": {
            "Max depth": "Controls the granularity of partitions.",
            "Min samples split": "Minimum samples required to split a node.",
        }
    },
    "Random Forest Regressor": {
        "how": "Averages predictions from many decision trees trained on bootstrap samples with random feature subsets.",
        "params": {
            "Trees": "Number of trees.",
            "Max depth": "Depth of individual trees.",
            "Max features": "Features considered per split.",
        }
    },
    "SVR": {
        "how": "Fits a function within an ε-tube around training data, ignoring small errors and penalizing larger ones.",
        "params": {
            "Kernel": "Determines the functional form of the regression surface.",
            "C": "Penalty for points outside the ε-tube.",
            "Gamma": "Kernel bandwidth for RBF/Poly.",
        }
    },
    "Gradient Boosting Regressor": {
        "how": "Builds trees sequentially, each fitting the residuals of the current ensemble.",
        "params": {
            "Estimators": "Number of boosting rounds.",
            "Learning rate": "Shrinks each tree's contribution.",
            "Max depth": "Depth of each weak learner.",
        }
    },
    "K-Means": {
        "how": "Alternates between assigning each point to its nearest centroid and updating centroids as cluster means until convergence.",
        "params": {
            "K": "Number of clusters.",
            "Init": "k-means++ seeds centroids intelligently to speed convergence; random is faster but less reliable.",
            "Max iterations": "Maximum number of assign-update cycles.",
        }
    },
    "K-Medians": {
        "how": "Like K-Means but uses the median instead of the mean for centroid updates, making it more robust to outliers.",
        "params": {
            "K": "Number of clusters.",
            "Max iterations": "Maximum iterations.",
        }
    },
    "K-Medoids": {
        "how": "Centers must be actual data points (medoids). More interpretable and robust than K-Means.",
        "params": {
            "K": "Number of clusters.",
            "Max iterations": "Maximum iterations.",
        }
    },
    "K-Modes": {
        "how": "Designed for categorical data. Uses the mode (most frequent value) instead of mean, and dissimilarity counts feature mismatches.",
        "params": {
            "K": "Number of clusters.",
            "Max iterations": "Maximum iterations.",
        }
    },
    "DBSCAN": {
        "how": "Groups points that are closely packed (core points), marks isolated points as noise, and labels border points to their nearest core cluster.",
        "params": {
            "Epsilon (ε)": "Neighborhood radius. Larger ε → fewer, bigger clusters.",
            "Min samples": "Minimum points in ε-neighborhood to be a core point.",
            "Metric": "Distance function for neighborhood calculation.",
        }
    },
    "Hierarchical": {
        "how": "Builds a tree of clusters (dendrogram) by merging the two closest clusters at each step. Cut the dendrogram at a level to get k clusters.",
        "params": {
            "N clusters": "Number of flat clusters to extract from the dendrogram.",
            "Linkage": "Ward minimizes intra-cluster variance. Complete uses maximum distance. Average uses average distance.",
            "Metric": "Distance function (Ward only supports euclidean).",
        }
    },
    "GMM": {
        "how": "Models data as a mixture of Gaussian distributions using the EM algorithm. Each cluster is described by its mean, covariance, and weight.",
        "params": {
            "Components": "Number of Gaussian components (clusters).",
            "Covariance type": "Full: each component has its own covariance. Tied: all share one. Diag/Spherical: simplified shapes.",
        }
    },
}
