Build a polished interactive **Machine Learning Algorithm Visualizer** that can be deployed on **GitHub Pages**.

The purpose is educational: users should be able to visually understand how ML algorithms behave when their important parameters change.

Do not use anything from my research.

## Core idea

Users can:

* Use built-in sample datasets
* Upload their own CSV dataset
* Select Classification, Regression, or Clustering
* Select an algorithm
* Change important algorithm parameters
* Train the model
* Visually inspect predictions, clusters, regression lines/surfaces, and decision boundaries
* Compare how changing parameters changes model behavior

Use **2D visualization wherever possible**.

If a dataset has more than 2 features, allow the user to:

* select any 2 features manually, OR
* project the data to 2D using PCA

Clearly indicate when PCA is being used.

---

# Datasets

Provide built-in datasets.

### Classification

* Iris
* Wine
* Breast Cancer
* Moons
* Circles

### Regression

* Synthetic linear regression
* Non-linear synthetic regression
* Diabetes dataset

### Clustering

* Blobs
* Moons
* Anisotropic clusters
* Noisy clusters

Also allow CSV upload.

For uploaded data:

* show columns
* let user select target column
* let user select feature columns
* detect classification/regression where possible
* allow manual override
* handle missing values with a simple strategy
* show a useful validation/error message for unsupported data

---

# Classification

Implement:

## KNN

Parameters:

* K
* Distance metric
* Weighting / voting method

Visualization:

* training points
* test points
* decision boundary
* highlight nearest neighbors when a test point is clicked
* optionally draw lines from selected point to its K neighbors
* show neighbor votes

This visualization should explain both **distance and voting**.

---

## Naive Bayes

Support:

* Gaussian Naive Bayes

Parameters where applicable:

* var_smoothing

Visualization:

* data points
* decision boundary
* class probability regions
* Gaussian distributions where useful

Show a small explanation of how posterior probabilities determine the prediction.

---

## Decision Tree

Parameters:

* criterion: Gini / Entropy
* max_depth
* min_samples_split

Visualization:

* decision boundary
* tree structure
* current depth
* class regions

Allow the user to compare Gini vs Entropy.

---

## Logistic Regression

Parameters:

* C / regularization strength
* penalty where supported
* max iterations

Visualization:

* decision boundary
* probability regions
* predicted probability for clicked points

For binary problems optionally visualize the logistic probability curve.

---

## SVM

Parameters:

* Kernel
* C
* Gamma

Support kernels:

* Linear
* Polynomial
* RBF
* Sigmoid where practical

Visualization:

* decision boundary
* margins
* support vectors
* classification regions

Changing kernel should visibly demonstrate how the boundary changes.

---

# Ensemble Methods

Implement:

## Bagging

Parameters:

* number of estimators
* sample fraction
* base estimator complexity

Visualization:

* individual estimator boundaries optionally
* combined boundary
* final prediction

---

## Random Forest

Parameters:

* number of trees
* max_depth
* max_features

Visualization:

* final decision boundary
* individual tree boundaries optionally
* feature importance
* prediction voting

---

## Gradient Boosting

Parameters:

* number of estimators
* learning rate
* max depth

Visualization:

* final decision boundary
* convergence / training score
* sequential improvement where practical

---

## AdaBoost

Parameters:

* number of estimators
* learning rate
* base estimator depth

Visualization:

* decision boundary
* misclassified point weights
* sequential learner contribution

---

## XGBoost

Parameters:

* number of estimators
* learning rate
* max depth

Use a browser-compatible implementation or implement a simplified educational version if full XGBoost cannot reasonably run entirely client-side.

Clearly label simplified implementations.

---

## LightGBM

Parameters:

* number of estimators
* learning rate
* number of leaves

If full LightGBM cannot run in the browser, implement a simplified educational approximation and clearly label it.

---

## CatBoost

Parameters:

* iterations
* learning rate
* depth

If full CatBoost cannot reasonably run client-side, use an educational approximation and clearly label it.

---

## Voting Classifier

Parameters:

* selected base models
* hard / soft voting
* voting weights

Visualization:

* boundaries of component models
* combined voting boundary
* individual model predictions

---

## Stacking

Parameters:

* selected base models
* meta-model
* number of folds

Visualization:

Show:

Base Models → Predictions → Meta Model → Final Prediction

Also visualize the resulting decision boundary.

---

## Classifier Chains

Support multilabel data where appropriate.

Parameters:

* base classifier
* chain order
* random state/order strategy

Use an appropriate synthetic multilabel dataset.

Show visually how previous predictions become inputs to later classifiers.

---

# Regression

Implement:

## Linear Regression

Show:

* scatter plot
* fitted regression line
* residual lines
* equation
* MSE
* R²

Also provide educational options for solving linear regression:

* analytical / normal equation
* gradient descent

For gradient descent allow parameters:

* learning rate
* iterations

Visualize how the regression line changes during optimization.

---

## Ridge Regression

Parameters:

* alpha

Visualization:

* fitted line
* coefficients
* coefficient shrinkage

---

## Lasso Regression

Parameters:

* alpha

Visualization:

* fitted line
* coefficients
* show coefficients becoming zero

---

## Elastic Net

Parameters:

* alpha
* L1 ratio

Visualization:

* fitted line
* coefficients

---

# Regression versions of ML algorithms

Where applicable also support regression variants:

* KNN Regressor
* Decision Tree Regressor
* Random Forest Regressor
* SVR
* Gradient Boosting Regressor

Use the same philosophy:

Expose a maximum of **3 important controlling parameters**.

---

# Unsupervised Learning

## K-Means

Parameters:

* K
* initialization
* max iterations

Visualization should support step-by-step execution:

1. initialize centroids
2. assign points
3. recompute centroids
4. repeat

Show centroid movement trails.

---

## K-Medians

Parameters:

* K
* initialization
* iterations

Visualize median-based center movement.

---

## K-Medoids

Parameters:

* K
* initialization
* iterations

Highlight actual observations used as medoids.

---

## K-Modes

Use categorical/sample encoded data where appropriate.

Parameters:

* K
* initialization
* iterations

Explain that K-Modes is designed for categorical features.

---

## DBSCAN

Parameters:

* epsilon
* min_samples
* distance metric

Visualization:

* clusters
* noise points
* core points
* border points
* epsilon neighborhood when clicking a point

---

## Hierarchical Clustering

Support:

* single linkage
* complete linkage
* average linkage
* Ward linkage

Parameters:

* number of clusters
* linkage method
* distance metric where compatible

Visualization:

* scatter plot
* clusters
* dendrogram

Allow the user to move the cluster cut level where feasible.

---

## Gaussian Mixture Model

Parameters:

* number of components
* covariance type
* initialization / iterations

Visualization:

* cluster assignments
* Gaussian ellipses
* component centers
* soft membership probability

---

# Parameter rule

For every algorithm expose a maximum of **3 important parameters**.

If an algorithm naturally has fewer important parameters, expose fewer.

Examples:

KNN:

* K
* distance metric
* voting weighting

SVM:

* kernel
* C
* gamma

Random Forest:

* trees
* max depth
* max features

DBSCAN:

* epsilon
* minimum samples
* distance metric

Do not expose dozens of advanced parameters.

The goal is **understanding, not complete API coverage**.

---

# Interaction

Provide:

* Train / Run
* Auto Run where step-based visualization makes sense
* Pause
* Next Step
* Reset
* Speed control

Step-by-step mode is especially important for:

* K-Means
* Gradient Descent
* Decision Trees where feasible
* Boosting
* KNN prediction
* hierarchical clustering

---

# Main Visualization

Use a large central interactive visualization.

Classification:

* scatter points
* decision boundary
* probability regions where useful

Regression:

* points
* prediction line / surface
* residuals

Clustering:

* points
* cluster colors
* centroids / medoids
* movement/history where applicable

Users should be able to hover/click points to inspect:

* actual value/class
* prediction
* probability
* neighbors
* cluster
* distance

depending on algorithm.

---

# Metrics

Show a compact metrics panel.

Classification:

* Accuracy
* Precision
* Recall
* F1
* Confusion Matrix

Regression:

* MAE
* MSE
* RMSE
* R²

Clustering:

* Silhouette score
* inertia where applicable
* number of clusters
* noise points for DBSCAN

Do not overwhelm the UI.

---

# Educational explanation

For every algorithm show a small panel:

### How it works

Explain the algorithm in 2–4 simple sentences.

### What the parameters mean

Explain the currently exposed parameters.

Example for SVM:

**C**
Controls how strongly the model penalizes classification errors.

**Gamma**
Controls how local the influence of each training point is.

**Kernel**
Controls the shape of the feature transformation / decision boundary.

Also dynamically explain visible effects when parameters change.

Example:

> Increasing K makes the KNN boundary smoother.

> Increasing tree depth creates more complex decision regions.

> Increasing gamma can make an RBF SVM boundary more local and complex.

---

# Architecture

Build as a static web application suitable for GitHub Pages.

Preferred:

* React
* TypeScript
* Vite
* Plotly.js / D3 / Canvas for visualization

No backend should be required.

For ML algorithms use:

* browser-compatible JavaScript ML libraries when appropriate
* clean TypeScript implementations for simple algorithms

Do not force Python or a server backend.

Keep:

* datasets
* algorithms
* metrics
* visualization
* UI components

separated into different modules.

---

# Suggested UI

Left sidebar:

Dataset
↓
Task
↓
Algorithm
↓
Parameters
↓
Train controls

Center:

Large interactive visualization

Right panel:

* algorithm explanation
* parameter explanation
* current model state
* metrics

Bottom:

Optional secondary visualization such as:

* convergence
* loss
* residuals
* confusion matrix
* tree
* dendrogram
* feature importance

depending on algorithm.

---

# Model comparison

Add a **Compare Mode**.

Allow users to compare up to 3 algorithms on the same dataset.

Example:

KNN vs Logistic Regression vs SVM

Show:

* decision boundaries side-by-side
* same train/test split
* same random seed
* key metrics

This is important for educational comparison.

---

# Reproducibility

Allow a random seed.

The same:

dataset + split + parameters + seed

should produce the same result.

---

# GitHub Pages

Configure deployment through GitHub Pages.

Include:

* GitHub Actions workflow
* correct Vite base URL
* README
* setup instructions
* deployment instructions

The app should work entirely from the GitHub Pages URL.

---

# Quality requirements

Create the complete application.

Before finishing:

* run TypeScript checks
* run linting
* run unit tests
* run production build
* fix all errors
* verify important algorithms manually
* verify CSV upload
* verify GitHub Pages routing
* verify responsive UI

`npm run build` must succeed.

Prioritize a clean educational visualization over implementing obscure algorithm features.
