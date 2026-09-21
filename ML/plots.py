import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage

# ─── palette ──────────────────────────────────────────────────────────────────

COLORS   = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED",
            "#DB2777", "#0891B2", "#65A30D"]
LIGHT    = ["rgba(37,99,235,0.18)", "rgba(220,38,38,0.18)",
            "rgba(5,150,105,0.18)", "rgba(217,119,6,0.18)",
            "rgba(124,58,237,0.18)", "rgba(219,39,119,0.18)"]
C_BEST   = "#D97706"
C_CONV   = "#059669"

_BASE = dict(
    template="simple_white",
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0",
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=12, r=12, t=44, b=12),
)

# ─── decision boundary helper ─────────────────────────────────────────────────

def _boundary_grid(model, X2d, n=160):
    lo = X2d.min(0) - 0.6
    hi = X2d.max(0) + 0.6
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], n),
                         np.linspace(lo[1], hi[1], n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    classes = np.unique(model.predict(X2d))  # to infer n_classes
    Z = model.predict(grid).reshape(xx.shape).astype(float)
    # normalize to [0, 1] for colorscale
    if Z.max() > 0:
        Z = Z / Z.max()
    return xx, yy, Z, lo, hi


def _discrete_cs(n):
    raw = [LIGHT[i % len(LIGHT)] for i in range(n)]
    if n == 1:
        return [[0, raw[0]], [1, raw[0]]]
    cs = []
    for i in range(n):
        cs += [[i / n, raw[i]], [(i + 1) / n - 1e-10, raw[i]]]
    cs[-1][0] = 1.0
    return cs

# ─── Classification ────────────────────────────────────────────────────────────

def plot_classification(X2d, y, model=None, feature_names=None,
                        class_names=None, title="",
                        highlight_idx=None, k_neighbors=5,
                        X_train=None, X_test=None, y_train=None, y_test=None):
    fn = feature_names or ["x₁", "x₂"]
    classes = np.unique(y)
    n_cls   = len(classes)
    cn      = class_names or [f"Class {c}" for c in classes]

    fig = go.Figure()

    # Background decision regions
    if model is not None:
        xx, yy, Z, lo, hi = _boundary_grid(model, X2d)
        cs = _discrete_cs(n_cls)
        fig.add_trace(go.Heatmap(
            x=np.linspace(lo[0], hi[0], xx.shape[1]),
            y=np.linspace(lo[1], hi[1], xx.shape[0]),
            z=Z, colorscale=cs, showscale=False, opacity=1.0,
            zmin=0, zmax=1,
        ))

    # Scatter: training points (if separate split provided)
    data_X = X2d if X_train is None else np.vstack([X_train, X_test])
    data_y = y   if y_train  is None else np.concatenate([y_train, y_test])
    train_n = len(X_train) if X_train is not None else len(X2d)

    is_tr = np.arange(len(data_y)) < train_n
    for i, cls in enumerate(classes):
        mask  = data_y == cls
        for train_flag, sym, edge, label_sfx in [
            (True,  "circle",         "white", " (train)"),
            (False, "diamond-open",   COLORS[i % len(COLORS)], " (test)"),
        ]:
            submask = mask & (is_tr if train_flag else ~is_tr)
            if not submask.any():
                continue
            fig.add_trace(go.Scatter(
                x=data_X[submask, 0], y=data_X[submask, 1],
                mode="markers", name=cn[i % len(cn)] + label_sfx,
                marker=dict(color=COLORS[i % len(COLORS)], size=7 if train_flag else 9,
                            symbol=sym, line=dict(color=edge, width=1)),
                showlegend=True,
            ))

    # KNN: highlight selected point and its neighbors
    if highlight_idx is not None and X_train is not None:
        pt = X_test[highlight_idx]
        fig.add_trace(go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers",
                                 name="Selected",
                                 marker=dict(color="gold", size=16, symbol="star",
                                             line=dict(color="black", width=1.5))))
        if model is not None and hasattr(model, "kneighbors"):
            nbrs_dist, nbrs_idx = model.kneighbors([pt], n_neighbors=k_neighbors)
            nbrs = X_train[nbrs_idx[0]]
            for nb in nbrs:
                fig.add_trace(go.Scatter(x=[pt[0], nb[0]], y=[pt[1], nb[1]],
                                         mode="lines", showlegend=False,
                                         line=dict(color="gold", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=nbrs[:, 0], y=nbrs[:, 1], mode="markers",
                                     name="Neighbors",
                                     marker=dict(color="gold", size=13, symbol="circle",
                                                 line=dict(color="black", width=1.5))))

    # SVM: support vectors
    if model is not None and hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        fig.add_trace(go.Scatter(x=sv[:, 0], y=sv[:, 1], mode="markers",
                                 name="Support Vectors",
                                 marker=dict(color="rgba(0,0,0,0)", size=14,
                                             symbol="circle-open",
                                             line=dict(color="#1E293B", width=2))))

    fig.update_layout(**_BASE, height=430,
                      title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#1E293B")),
                      xaxis=dict(title=fn[0], showgrid=True, gridcolor="#F1F5F9"),
                      yaxis=dict(title=fn[1], showgrid=True, gridcolor="#F1F5F9"))
    return fig

# ─── Regression ────────────────────────────────────────────────────────────────

def plot_regression(X_train, y_train, X_test, y_test, model,
                    feature_name="x", target_name="y", title="",
                    show_residuals=True):
    # For 1D: sort for smooth line
    is_1d = X_train.shape[1] == 1
    fig = go.Figure()

    if is_1d:
        x_range = np.linspace(
            min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5,
            max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5,
            400).reshape(-1, 1)
        y_line = model.predict(x_range)
        fig.add_trace(go.Scatter(x=x_range[:, 0], y=y_line,
                                 mode="lines", name="Model fit",
                                 line=dict(color=C_BEST, width=2.5)))

    # Training scatter
    fig.add_trace(go.Scatter(
        x=X_train[:, 0], y=y_train, mode="markers", name="Train",
        marker=dict(color=COLORS[0], size=7, line=dict(color="white", width=0.5))))
    # Test scatter
    y_pred_test = model.predict(X_test)
    fig.add_trace(go.Scatter(
        x=X_test[:, 0], y=y_test, mode="markers", name="Test",
        marker=dict(color=COLORS[1], size=9, symbol="diamond",
                    line=dict(color="white", width=0.5))))

    # Residuals for test set (1D only)
    if show_residuals and is_1d:
        for xi, ytrue, ypred in zip(X_test[:, 0], y_test, y_pred_test):
            fig.add_shape(type="line", x0=xi, y0=ytrue, x1=xi, y1=ypred,
                          line=dict(color="rgba(220,38,38,0.4)", width=1.5))

    fig.update_layout(**_BASE, height=430,
                      title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#1E293B")),
                      xaxis=dict(title=feature_name, showgrid=True, gridcolor="#F1F5F9"),
                      yaxis=dict(title=target_name,  showgrid=True, gridcolor="#F1F5F9"))
    return fig

# ─── Clustering ────────────────────────────────────────────────────────────────

def plot_clustering(X2d, labels, centroids=None, trails=None,
                    feature_names=None, title="", dbscan_types=None):
    fn = feature_names or ["x₁", "x₂"]
    fig = go.Figure()

    unique = np.unique(labels)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        if lbl == -1:   # DBSCAN noise
            name, col, sym = "Noise", "#94A3B8", "x"
        else:
            name, col, sym = f"Cluster {lbl}", COLORS[i % len(COLORS)], "circle"

        # point types for DBSCAN
        if dbscan_types is not None:
            sizes = np.where(dbscan_types[mask] == "core", 9,
                    np.where(dbscan_types[mask] == "border", 7, 5))
        else:
            sizes = 8

        fig.add_trace(go.Scatter(
            x=X2d[mask, 0], y=X2d[mask, 1], mode="markers", name=name,
            marker=dict(color=col, size=sizes, symbol=sym,
                        line=dict(color="white", width=0.6))))

    # Centroid trails
    if trails is not None:
        for k_i, trail in enumerate(trails):
            trail = np.array(trail)
            if len(trail) < 2:
                continue
            fig.add_trace(go.Scatter(x=trail[:, 0], y=trail[:, 1],
                                     mode="lines", name=f"Trail {k_i}",
                                     line=dict(color=COLORS[k_i % len(COLORS)],
                                               width=1.5, dash="dot"),
                                     showlegend=False))

    # Centroids / medoids
    if centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode="markers",
            name="Centroids",
            marker=dict(color=[COLORS[i % len(COLORS)] for i in range(len(centroids))],
                        size=18, symbol="star",
                        line=dict(color="#1E293B", width=1.5))))

    # GMM ellipses via shapes
    fig.update_layout(**_BASE, height=430,
                      title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#1E293B")),
                      xaxis=dict(title=fn[0], showgrid=True, gridcolor="#F1F5F9"),
                      yaxis=dict(title=fn[1], showgrid=True, gridcolor="#F1F5F9"))
    return fig


def add_gmm_ellipses(fig, gmm, n_std=2.0):
    """Overlay GMM covariance ellipses on a clustering figure."""
    for i, (mean, cov_type) in enumerate(zip(gmm.means_, _iter_covs(gmm))):
        vals, vecs = np.linalg.eigh(cov_type)
        order  = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h   = 2 * n_std * np.sqrt(np.abs(vals))
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = w / 2 * np.cos(t)
        ellipse_y = h / 2 * np.sin(t)
        a_rad = np.radians(angle)
        rx = ellipse_x * np.cos(a_rad) - ellipse_y * np.sin(a_rad) + mean[0]
        ry = ellipse_x * np.sin(a_rad) + ellipse_y * np.cos(a_rad) + mean[1]
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", showlegend=False,
                                 line=dict(color=COLORS[i % len(COLORS)], width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode="markers",
                                 showlegend=False,
                                 marker=dict(color=COLORS[i % len(COLORS)], size=10,
                                             symbol="cross", line=dict(color="black", width=1))))
    return fig


def _iter_covs(gmm):
    if gmm.covariance_type == "full":
        return gmm.covariances_
    if gmm.covariance_type == "tied":
        return [gmm.covariances_] * gmm.n_components
    if gmm.covariance_type == "diag":
        return [np.diag(c) for c in gmm.covariances_]
    return [np.eye(gmm.means_.shape[1]) * c for c in gmm.covariances_]

# ─── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names):
    fig = go.Figure(go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
        showscale=False,
        text=cm.astype(str), texttemplate="%{text}",
        textfont=dict(size=14, color="#1E293B"),
    ))
    fig.update_layout(**_BASE, height=300,
                      title=dict(text="<b>Confusion Matrix</b>",
                                 font=dict(size=13, color="#1E293B")),
                      xaxis=dict(title="Predicted"),
                      yaxis=dict(title="Actual", autorange="reversed"))
    return fig

# ─── Dendrogram ────────────────────────────────────────────────────────────────

def plot_dendrogram_fig(X, method="ward", n_clusters=3):
    Z   = linkage(X, method=method)
    dn  = dendrogram(Z, no_plot=True, truncate_mode="lastp", p=30)
    fig = go.Figure()
    for xs, ys in zip(dn["icoord"], dn["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                 line=dict(color=COLORS[0], width=1.5)))
    # Cut line
    from scipy.cluster.hierarchy import fcluster
    cut_h = _cut_height(Z, n_clusters)
    fig.add_hline(y=cut_h, line=dict(color=COLORS[1], width=2, dash="dash"),
                  annotation_text=f"Cut: {n_clusters} clusters",
                  annotation_position="top right")
    fig.update_layout(**_BASE, height=280,
                      title=dict(text="<b>Dendrogram</b>", font=dict(size=13, color="#1E293B")),
                      xaxis=dict(visible=False),
                      yaxis=dict(title="Distance"))
    return fig


def _cut_height(Z, n_clusters):
    n = Z.shape[0] + 1
    if n_clusters >= n:
        return 0
    return float(Z[-(n_clusters - 1), 2]) * 0.999 if n_clusters > 1 else Z[-1, 2]

# ─── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(importances, feature_names, title="Feature Importance"):
    order = np.argsort(importances)
    fig   = go.Figure(go.Bar(
        x=importances[order], y=[feature_names[i] for i in order],
        orientation="h", marker_color=COLORS[0],
    ))
    fig.update_layout(**_BASE, height=max(180, 25 * len(feature_names)),
                      title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#1E293B")),
                      xaxis=dict(title="Importance"), yaxis=dict(title=""),
                      margin=dict(l=120, r=12, t=44, b=12))
    return fig

# ─── Regression coefficients ───────────────────────────────────────────────────

def plot_coefficients(coef, feature_names, title="Coefficients"):
    colors = [COLORS[0] if c >= 0 else COLORS[1] for c in coef]
    order  = np.argsort(np.abs(coef))
    fig    = go.Figure(go.Bar(
        x=coef[order], y=[feature_names[i] for i in order],
        orientation="h", marker_color=[colors[i] for i in order],
    ))
    fig.update_layout(**_BASE, height=max(180, 25 * len(feature_names)),
                      title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#1E293B")),
                      xaxis=dict(title="Coefficient value"), yaxis=dict(title=""),
                      margin=dict(l=120, r=12, t=44, b=12))
    return fig

# ─── Convergence / training scores ────────────────────────────────────────────

def plot_convergence(scores, title="Training Score vs Estimators"):
    fig = go.Figure(go.Scatter(y=scores, mode="lines+markers",
                               line=dict(color=C_CONV, width=2),
                               marker=dict(size=5, color=C_CONV),
                               fill="tozeroy", fillcolor="rgba(5,150,105,0.08)"))
    fig.update_layout(**_BASE, height=165,
                      title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#1E293B")),
                      xaxis=dict(title="Estimator #", showgrid=True, gridcolor="#F1F5F9"),
                      yaxis=dict(title="Score", showgrid=True, gridcolor="#F1F5F9"),
                      showlegend=False)
    return fig

# ─── Compare: side-by-side ────────────────────────────────────────────────────

def plot_compare(results, task):
    """results: list of dicts with keys fig, algo_name, metrics."""
    n = len(results)
    titles = [r["algo"] for r in results]
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles,
                        shared_yaxes=True)
    for ci, r in enumerate(results, 1):
        for trace in r["fig"].data:
            fig.add_trace(trace, row=1, col=ci)
    fig.update_layout(**_BASE, height=430,
                      title=dict(text="<b>Algorithm Comparison</b>",
                                 font=dict(size=14, color="#1E293B")))
    return fig
