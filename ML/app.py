import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as sk_metrics
from sklearn.tree import export_text

from datasets import (
    get_dataset, CLF_DATASETS, REG_DATASETS, CLUST_DATASETS,
    CLF_ALGOS, REG_ALGOS, CLUST_ALGOS, STEP_ALGOS,
    process_csv, load_multilabel, make_kmodes_data,
)
from models import (
    make_classifier, make_regressor, make_cluster_model,
    kmeans_steps, kmedians_steps, kmedoids_steps, kmodes_steps,
    DESCRIPTIONS,
)
from plots import (
    plot_classification, plot_regression, plot_clustering,
    plot_confusion_matrix, plot_dendrogram_fig, plot_feature_importance,
    plot_coefficients, plot_convergence, plot_actual_vs_predicted,
    add_gmm_ellipses, COLORS,
)

# ─── page config ──────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="ML Visualizer", page_icon="🤖")

# ─── CSS (same look as Optimization app) ──────────────────────────────────────

st.markdown("""
<style>
.block-container { padding: 0.8rem 1.2rem 0.5rem 1.2rem !important; }
[data-testid="stHeader"]     { background: transparent !important; border: none !important; box-shadow: none !important; }
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

.ctrl-panel {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 1rem 0.9rem;
}
.ctrl-section {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: #94A3B8; margin: 0.9rem 0 0.3rem 0;
}
.stats-panel {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 0.9rem;
}
[data-testid="metric-container"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 0.3rem 0.6rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 4px;
}
[data-testid="stMetricValue"] { color: #1E293B !important; font-size: 1rem !important; }
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.7rem !important; }
.stButton > button { border-radius: 8px !important; font-weight: 600 !important; border: 1px solid #CBD5E1 !important; }
.plot-box { border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ─── session state ────────────────────────────────────────────────────────────

for k, v in {
    "result": None, "steps": None, "step_idx": 0,
    "is_running": False, "run_key": "", "csv_data": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(135deg,#1E40AF 0%,#3B82F6 100%);
            padding:0.85rem 1.3rem;border-radius:12px;margin-bottom:0.7rem;">
  <span style="color:white;font-size:1.4rem;font-weight:700;">🤖 ML Algorithm Visualizer</span><br>
  <span style="color:#BFDBFE;font-size:0.82rem;">
    Interactively explore how machine learning algorithms work and how parameters affect results
  </span>
</div>
""", unsafe_allow_html=True)

# ─── 3-column layout ──────────────────────────────────────────────────────────

ctrl_col, plot_col, info_col = st.columns([1.15, 2.9, 1.15])

# ══════════════════════ LEFT: controls ════════════════════════════════════════

with ctrl_col:
    st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)

    # ── Task ──
    st.markdown('<div class="ctrl-section">Task</div>', unsafe_allow_html=True)
    task = st.selectbox("Task", ["Classification", "Regression", "Clustering"],
                        label_visibility="collapsed")

    # ── Dataset ──
    st.markdown('<div class="ctrl-section">Dataset</div>', unsafe_allow_html=True)
    ds_list = {"Classification": CLF_DATASETS,
               "Regression":     REG_DATASETS,
               "Clustering":     CLUST_DATASETS}[task]
    ds_name = st.selectbox("Dataset", ds_list, label_visibility="collapsed")

    # CSV upload
    X_raw = y_raw = feat_names = class_names = None
    if ds_name == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
            all_cols   = list(df.columns)
            target_col = st.selectbox("Target column", all_cols, index=len(all_cols)-1)
            feat_cols  = st.multiselect("Feature columns",
                                        [c for c in all_cols if c != target_col],
                                        default=[c for c in all_cols if c != target_col])
            missing    = st.selectbox("Handle missing values", ["Drop rows", "Fill mean"])
            if feat_cols:
                X_raw, y_raw, feat_names, class_names = process_csv(
                    df, target_col, feat_cols, task, missing)
    else:
        # built-in
        if task == "Clustering":
            result = get_dataset(task, ds_name)
            X_raw, y_raw, feat_names, class_names = result[0], result[1], result[2], None
        else:
            X_raw, y_raw, feat_names, class_names = get_dataset(task, ds_name)

    # ── Algorithm ──
    st.markdown('<div class="ctrl-section">Algorithm</div>', unsafe_allow_html=True)
    algo_list = {"Classification": CLF_ALGOS,
                 "Regression":     REG_ALGOS,
                 "Clustering":     CLUST_ALGOS}[task]
    algo = st.selectbox("Algorithm", algo_list, label_visibility="collapsed")

    # Classifier chain needs multilabel
    if algo == "Classifier Chain":
        X_raw, y_raw, feat_names, class_names = load_multilabel()
        st.caption("ℹ️ Classifier Chain uses the built-in multilabel dataset.")

    # K-Modes needs categorical data
    if algo == "K-Modes":
        X_raw, y_raw, feat_names = make_kmodes_data()
        class_names = None
        st.caption("ℹ️ K-Modes uses integer-encoded categorical data.")

    # ── Feature reduction for >2D ──
    use_pca = False; feat_idx = (0, 1)
    if X_raw is not None and X_raw.ndim == 2 and X_raw.shape[1] > 2:
        st.markdown('<div class="ctrl-section">2D Projection</div>', unsafe_allow_html=True)
        proj = st.radio("Project to 2D via", ["PCA", "Select 2 features"], horizontal=True)
        if proj == "Select 2 features":
            f1 = st.selectbox("Feature 1", feat_names, index=0)
            f2 = st.selectbox("Feature 2", feat_names, index=1)
            feat_idx = (feat_names.index(f1), feat_names.index(f2))
        else:
            use_pca = True

    # ── Algorithm Parameters (max 3) ──
    st.markdown('<div class="ctrl-section">Parameters</div>', unsafe_allow_html=True)
    p = {}

    # ── Classification params ──
    if algo == "KNN":
        p["K"]       = st.slider("K (neighbors)", 1, 30, 5)
        p["metric"]  = st.selectbox("Distance metric", ["euclidean", "manhattan", "chebyshev"])
        p["weights"] = st.selectbox("Voting", ["uniform", "distance"])
    elif algo == "Naive Bayes":
        p["var_smoothing"] = st.select_slider("Var smoothing",
            options=[1e-12, 1e-10, 1e-9, 1e-7, 1e-5, 1e-3, 0.01], value=1e-9,
            format_func=lambda x: f"{x:.0e}")
    elif algo == "Decision Tree":
        p["criterion"]        = st.selectbox("Criterion", ["Gini", "Entropy"])
        p["max_depth"]        = st.slider("Max depth (0 = unlimited)", 0, 20, 5)
        p["min_samples_split"]= st.slider("Min samples split", 2, 20, 2)
    elif algo == "Logistic Regression":
        p["C"]        = st.slider("C (regularization)", 0.01, 20.0, 1.0, 0.01)
        p["penalty"]  = st.selectbox("Penalty", ["l2", "l1"])
        p["max_iter"] = st.slider("Max iterations", 100, 2000, 500, 100)
    elif algo == "SVM":
        p["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        p["C"]      = st.slider("C", 0.01, 20.0, 1.0, 0.01)
        p["gamma"]  = st.selectbox("Gamma", ["scale", "auto", "0.1", "0.5", "1.0", "5.0"])
    elif algo == "Bagging":
        p["n_estimators"]    = st.slider("Estimators", 5, 100, 10)
        p["sample_fraction"] = st.slider("Sample fraction", 0.3, 1.0, 0.8, 0.05)
        p["base_depth"]      = st.slider("Base tree depth", 1, 10, 3)
    elif algo in ("Random Forest", "Random Forest Regressor"):
        p["n_estimators"] = st.slider("Trees", 10, 200, 50, 10)
        p["max_depth"]    = st.slider("Max depth (0 = unlimited)", 0, 20, 0)
        p["max_features"] = st.selectbox("Max features", ["sqrt", "log2"])
    elif algo in ("Gradient Boosting", "Gradient Boosting Regressor"):
        p["n_estimators"] = st.slider("Estimators", 10, 300, 100, 10)
        p["lr"]           = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
        p["max_depth"]    = st.slider("Max depth", 1, 10, 3)
    elif algo == "AdaBoost":
        p["n_estimators"] = st.slider("Estimators", 10, 200, 50, 10)
        p["lr"]           = st.slider("Learning rate", 0.01, 2.0, 1.0, 0.01)
        p["base_depth"]   = st.slider("Base stump depth", 1, 5, 1)
    elif algo in ("XGBoost", "LightGBM"):
        p["n_estimators"] = st.slider("Estimators", 10, 300, 100, 10)
        p["lr"]           = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        if algo == "LightGBM":
            p["num_leaves"] = st.slider("Num leaves", 10, 100, 31)
        else:
            p["max_depth"] = st.slider("Max depth", 1, 10, 3)
    elif algo == "CatBoost (approx.)":
        p["iterations"] = st.slider("Iterations", 10, 300, 100, 10)
        p["lr"]         = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        p["depth"]      = st.slider("Depth", 1, 10, 3)
    elif algo == "Voting Classifier":
        p["voting"] = st.selectbox("Voting", ["soft", "hard"])
    elif algo == "Stacking":
        p["cv_folds"] = st.slider("CV folds", 3, 10, 5)
    elif algo == "Classifier Chain":
        p["order"] = st.selectbox("Chain order", ["random"])

    # ── Regression params ──
    elif algo in ("Ridge", "Lasso"):
        p["alpha"] = st.slider("Alpha (regularization)", 0.001, 10.0, 1.0, 0.001, format="%.3f")
    elif algo == "Elastic Net":
        p["alpha"]    = st.slider("Alpha", 0.001, 10.0, 1.0, 0.001, format="%.3f")
        p["l1_ratio"] = st.slider("L1 ratio (0=Ridge, 1=Lasso)", 0.0, 1.0, 0.5, 0.05)
    elif algo == "KNN Regressor":
        p["K"]       = st.slider("K (neighbors)", 1, 30, 5)
        p["metric"]  = st.selectbox("Distance metric", ["euclidean", "manhattan"])
        p["weights"] = st.selectbox("Voting", ["uniform", "distance"])
    elif algo == "Decision Tree Regressor":
        p["max_depth"]         = st.slider("Max depth (0 = unlimited)", 0, 20, 5)
        p["min_samples_split"] = st.slider("Min samples split", 2, 20, 2)
    elif algo == "SVR":
        p["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        p["C"]      = st.slider("C", 0.01, 20.0, 1.0, 0.01)
        p["gamma"]  = st.selectbox("Gamma", ["scale", "auto", "0.1", "1.0"])

    # ── Clustering params ──
    elif algo in ("K-Means", "K-Medians", "K-Medoids"):
        p["K"]        = st.slider("K (clusters)", 2, 10, 3)
        if algo == "K-Means":
            p["init"] = st.selectbox("Initialization", ["k-means++", "random"])
        p["max_iter"] = st.slider("Max iterations", 5, 100, 30)
    elif algo == "K-Modes":
        p["K"]        = st.slider("K (clusters)", 2, 8, 3)
        p["max_iter"] = st.slider("Max iterations", 5, 50, 20)
    elif algo == "DBSCAN":
        p["eps"]        = st.slider("Epsilon (ε)", 0.05, 3.0, 0.5, 0.05)
        p["min_samples"]= st.slider("Min samples", 2, 20, 5)
        p["metric"]     = st.selectbox("Metric", ["euclidean", "manhattan"])
    elif algo == "Hierarchical":
        p["n_clusters"] = st.slider("N clusters", 2, 10, 3)
        p["linkage"]    = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
    elif algo == "GMM":
        p["n_components"] = st.slider("Components", 2, 10, 3)
        p["cov_type"]     = st.selectbox("Covariance type", ["full", "tied", "diag", "spherical"])

    # ── Global settings ──
    st.markdown('<div class="ctrl-section">Settings</div>', unsafe_allow_html=True)
    seed      = int(st.number_input("Random seed", value=42, min_value=0, step=1))
    if task in ("Classification", "Regression"):
        test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05)

    # ── Compare mode ──
    st.markdown('<div class="ctrl-section">Compare</div>', unsafe_allow_html=True)
    compare_mode = st.checkbox("Compare Mode (2 algorithms)")
    algo2 = None
    if compare_mode:
        algo2 = st.selectbox("Algorithm 2", algo_list,
                             index=min(1, len(algo_list)-1), label_visibility="collapsed")
        p2 = {}   # use defaults for algo2

    # ── Buttons ──
    st.markdown('<div class="ctrl-section">Controls</div>', unsafe_allow_html=True)
    is_step = algo in STEP_ALGOS

    if is_step:
        b1, b2, b3, b4 = st.columns(4)
        run_btn   = b1.button("▶", use_container_width=True, type="primary")
        pause_btn = b2.button("⏸", use_container_width=True)
        step_btn  = b3.button("⏭", use_container_width=True)
        reset_btn = b4.button("↺", use_container_width=True)
        speed_val = st.slider("Speed", 1, 10, 6)
        delay     = round(0.8 * (0.01 / 0.8) ** ((speed_val - 1) / 9), 3)
    else:
        train_btn = st.button("🚀 Train", type="primary", use_container_width=True)
        reset_btn = st.button("↺  Reset", use_container_width=True)
        run_btn = pause_btn = step_btn = False

    st.markdown('</div>', unsafe_allow_html=True)

# ─── config key → auto-reset on change ───────────────────────────────────────

cfg_key = f"{task}|{ds_name}|{algo}|{seed}"
if st.session_state.run_key != cfg_key:
    st.session_state.result     = None
    st.session_state.steps      = None
    st.session_state.step_idx   = 0
    st.session_state.is_running = False
    st.session_state.run_key    = cfg_key

# ─── Data preparation helper ──────────────────────────────────────────────────

def prepare_2d(X, feat_names):
    """Return (X2d, feat_names_2d, pca_obj_or_None)."""
    if X is None or X.shape[1] <= 2:
        return X, feat_names, None
    if use_pca:
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        pca    = PCA(n_components=2, random_state=seed)
        X2d    = pca.fit_transform(X_sc)
        var    = pca.explained_variance_ratio_
        names  = [f"PC1 ({var[0]:.0%})", f"PC2 ({var[1]:.0%})"]
        return X2d, names, pca
    else:
        i, j = feat_idx
        return X[:, [i, j]], [feat_names[i], feat_names[j]], None

# ─── Training / step-init logic ───────────────────────────────────────────────

def do_train():
    if X_raw is None:
        return
    X2d, fn2d, pca_obj = prepare_2d(X_raw, feat_names)

    if task == "Clustering":
        model = make_cluster_model(algo, p, seed)
        if model is None:
            return   # handled by step-based branch
        if algo == "GMM":
            labels = model.fit_predict(X2d)
        else:
            labels = model.fit_predict(X2d)
        centroids = getattr(model, "cluster_centers_", None)
        if centroids is None and algo == "GMM":
            centroids = model.means_
        st.session_state.result = dict(
            task=task, algo=algo, X2d=X2d, labels=labels,
            model=model, feature_names=fn2d, centroids=centroids,
        )
        return

    # Supervised
    is_multilabel = y_raw.ndim == 2
    X_tr, X_te, y_tr, y_te = train_test_split(
        X2d, y_raw, test_size=test_size, random_state=seed,
        stratify=y_raw if task == "Classification" and not is_multilabel else None)

    if task == "Classification":
        model = make_classifier(algo, p, seed)
    else:
        model = make_regressor(algo, p, seed)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

    st.session_state.result = dict(
        task=task, algo=algo,
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        X2d=np.vstack([X_tr, X_te]), y_all=np.concatenate([y_tr, y_te])
        if not is_multilabel else y_raw,
        model=model, y_pred=y_pred, y_prob=y_prob,
        feature_names=fn2d, class_names=class_names,
        is_multilabel=is_multilabel, pca=pca_obj,
    )


def init_steps():
    if X_raw is None:
        return
    X2d, fn2d, _ = prepare_2d(X_raw, feat_names)
    k  = p.get("K", 3)
    mx = p.get("max_iter", 30)
    if algo == "K-Means":
        steps = kmeans_steps(X2d, k, p.get("init", "k-means++"), mx, seed)
    elif algo == "K-Medians":
        steps = kmedians_steps(X2d, k, mx, seed)
    elif algo == "K-Medoids":
        steps = kmedoids_steps(X2d, k, mx, seed)
    elif algo == "K-Modes":
        steps = kmodes_steps(X2d.astype(int), k, mx, seed)
    else:
        return
    st.session_state.steps      = steps
    st.session_state.step_idx   = 0
    st.session_state.result     = dict(
        task="Clustering", algo=algo,
        X2d=X2d, feature_names=fn2d)

# Handle button clicks
if is_step:
    if reset_btn:
        st.session_state.steps      = None
        st.session_state.step_idx   = 0
        st.session_state.is_running = False
        st.session_state.result     = None
    if pause_btn:
        st.session_state.is_running = False
    if run_btn:
        if st.session_state.steps is None:
            init_steps()
        st.session_state.is_running = True
    if step_btn:
        st.session_state.is_running = False
        if st.session_state.steps is None:
            init_steps()
        elif st.session_state.step_idx < len(st.session_state.steps) - 1:
            st.session_state.step_idx += 1
else:
    if reset_btn:
        st.session_state.result     = None
        st.session_state.is_running = False
    if train_btn:
        do_train()

res   = st.session_state.result
steps = st.session_state.steps

# ══════════════════════ CENTRE: visualization ═════════════════════════════════

with plot_col:

    if is_step:
        n_steps = len(steps) if steps else 1
        cur     = st.session_state.step_idx
        status  = "🟢 Running" if st.session_state.is_running else \
                  ("✅ Converged" if steps and cur >= len(steps)-1 else "⏸ Paused")
        st.progress(cur / max(n_steps-1, 1),
                    text=f"{status}  —  step {cur+1} / {n_steps}" if steps else "Press ▶ to start")
    else:
        if res:
            st.progress(1.0, text="✅ Trained")
        else:
            st.progress(0.0, text="Press 🚀 Train to start")

    st.markdown('<div class="plot-box">', unsafe_allow_html=True)

    if res is None and steps is None:
        st.info("Configure settings on the left and press **🚀 Train** (or ▶ for step-based).",
                icon="💡")

    # ── Step-based clustering ──
    elif is_step and steps is not None:
        idx      = st.session_state.step_idx
        cens, lbs = steps[idx]
        X2d      = res["X2d"]
        fn2d     = res["feature_names"]
        k        = p.get("K", 3)
        # Centroid trails
        trails   = [[steps[t][0][i] for t in range(idx+1)] for i in range(k)]
        fig      = plot_clustering(X2d, lbs, centroids=cens, trails=trails,
                                   feature_names=fn2d,
                                   title=f"{algo}  —  Step {idx+1} / {len(steps)}")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Compare mode for step-based clustering
        if compare_mode and algo2:
            st.divider()
            st.caption(f"**{algo2}** (default params)")
            lbs2, cens2, m2 = None, None, None

            if algo2 in STEP_ALGOS:
                k2, mx2 = 3, 30
                if algo2 == "K-Means":
                    steps2 = kmeans_steps(X2d, k2, "k-means++", mx2, seed)
                elif algo2 == "K-Medians":
                    steps2 = kmedians_steps(X2d, k2, mx2, seed)
                elif algo2 == "K-Medoids":
                    steps2 = kmedoids_steps(X2d, k2, mx2, seed)
                elif algo2 == "K-Modes":
                    steps2 = kmodes_steps(X2d.astype(int), k2, 20, seed)
                final2 = steps2[-1]
                lbs2, cens2 = final2["labels"], final2.get("centroids")
            else:
                m2    = make_cluster_model(algo2, {}, seed)
                lbs2  = m2.fit_predict(X2d)
                cens2 = getattr(m2, "cluster_centers_", None)
                if cens2 is None and algo2 == "GMM":
                    cens2 = m2.means_

            db_types2 = None
            if algo2 == "DBSCAN" and m2 is not None:
                core2 = set(m2.core_sample_indices_)
                db_types2 = np.array(
                    ["core" if i in core2 else ("border" if lbs2[i] != -1 else "noise")
                     for i in range(len(lbs2))])

            fig2 = plot_clustering(X2d, lbs2, centroids=cens2,
                                   feature_names=fn2d,
                                   title=f"{algo2}  ·  {ds_name}",
                                   dbscan_types=db_types2)
            if algo2 == "GMM" and m2 is not None:
                fig2 = add_gmm_ellipses(fig2, m2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Classification ──
    elif res and res["task"] == "Classification":
        r = res
        is_ml = r.get("is_multilabel", False)
        if is_ml:
            y_display = r["y_test"][:, 0]   # show first label for viz
        else:
            y_display = r["y_test"]

        # KNN: select a test point for neighbor viz
        hl_idx  = None
        if algo == "KNN":
            hl_idx = st.slider("Highlight test point", 0, len(r["X_test"])-1, 0)

        fig = plot_classification(
            X2d=np.vstack([r["X_train"], r["X_test"]]),
            y=np.concatenate([r["y_train"], r["y_test"] if not is_ml
                              else r["y_test"][:, 0]]),
            model=r["model"],
            feature_names=r["feature_names"],
            class_names=r["class_names"],
            title=f"{algo}  ·  {ds_name}",
            highlight_idx=hl_idx,
            k_neighbors=p.get("K", 5),
            X_train=r["X_train"], X_test=r["X_test"],
            y_train=r["y_train"],
            y_test=y_display,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Compare mode: show second algorithm
        if compare_mode and algo2 and not is_ml:
            st.divider()
            st.caption(f"**{algo2}** (default params)")
            m2 = make_classifier(algo2, {
                "K":5,"metric":"euclidean","weights":"uniform",
                "var_smoothing":1e-9,"criterion":"Gini","max_depth":5,
                "min_samples_split":2,"C":1.0,"penalty":"l2","max_iter":500,
                "kernel":"rbf","gamma":"scale","n_estimators":50,"lr":0.1,
                "base_depth":1,"max_features":"sqrt","num_leaves":31,
                "iterations":50,"depth":3,"voting":"soft","cv_folds":5,"order":"random",
                "sample_fraction":0.8,
            }, seed)
            m2.fit(r["X_train"], r["y_train"])
            y_pred2 = m2.predict(r["X_test"])
            fig2 = plot_classification(
                X2d=np.vstack([r["X_train"], r["X_test"]]),
                y=np.concatenate([r["y_train"], r["y_test"]]),
                model=m2, feature_names=r["feature_names"],
                class_names=r["class_names"],
                title=f"{algo2}  ·  {ds_name}",
                X_train=r["X_train"], X_test=r["X_test"],
                y_train=r["y_train"], y_test=r["y_test"],
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            acc2 = sk_metrics.accuracy_score(r["y_test"], y_pred2)
            st.caption(f"{algo2} test accuracy: **{acc2:.3f}**")

    # ── Regression ──
    elif res and res["task"] == "Regression":
        r = res
        fig = plot_regression(
            r["X_train"], r["y_train"],
            r["X_test"],  r["y_test"],
            r["model"],
            feature_name=r["feature_names"][0] if r["feature_names"] else "x",
            title=f"{algo}  ·  {ds_name}",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Compare mode: second regression algorithm
        if compare_mode and algo2:
            st.divider()
            st.caption(f"**{algo2}** (default params)")
            m2 = make_regressor(algo2, {}, seed)
            m2.fit(r["X_train"], r["y_train"])
            y_pred2 = m2.predict(r["X_test"])
            fig2 = plot_regression(
                r["X_train"], r["y_train"],
                r["X_test"],  r["y_test"],
                m2,
                feature_name=r["feature_names"][0] if r["feature_names"] else "x",
                title=f"{algo2}  ·  {ds_name}",
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            r2_score = sk_metrics.r2_score(r["y_test"], y_pred2)
            st.caption(f"{algo2} test R²: **{r2_score:.3f}**")

    # ── Non-step Clustering ──
    elif res and res["task"] == "Clustering":
        r    = res
        lbs  = r["labels"]
        X2d  = r["X2d"]
        cens = r.get("centroids")

        # DBSCAN: classify core / border / noise
        db_types = None
        if algo == "DBSCAN":
            model  = r["model"]
            core   = set(model.core_sample_indices_)
            db_types = np.array(
                ["core" if i in core else ("border" if lbs[i] != -1 else "noise")
                 for i in range(len(lbs))])

        fig = plot_clustering(X2d, lbs, centroids=cens,
                              feature_names=r["feature_names"],
                              title=f"{algo}  ·  {ds_name}",
                              dbscan_types=db_types)
        if algo == "GMM":
            fig = add_gmm_ellipses(fig, r["model"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Compare mode: second clustering algorithm
        if compare_mode and algo2:
            st.divider()
            st.caption(f"**{algo2}** (default params)")
            lbs2, cens2, m2 = None, None, None

            if algo2 in STEP_ALGOS:
                k2, mx2 = 3, 30
                if algo2 == "K-Means":
                    steps2 = kmeans_steps(X2d, k2, "k-means++", mx2, seed)
                elif algo2 == "K-Medians":
                    steps2 = kmedians_steps(X2d, k2, mx2, seed)
                elif algo2 == "K-Medoids":
                    steps2 = kmedoids_steps(X2d, k2, mx2, seed)
                elif algo2 == "K-Modes":
                    steps2 = kmodes_steps(X2d.astype(int), k2, 20, seed)
                final2 = steps2[-1]
                lbs2  = final2["labels"]
                cens2 = final2.get("centroids")
            else:
                m2   = make_cluster_model(algo2, {}, seed)
                lbs2 = m2.fit_predict(X2d)
                cens2 = getattr(m2, "cluster_centers_", None)
                if cens2 is None and algo2 == "GMM":
                    cens2 = m2.means_

            db_types2 = None
            if algo2 == "DBSCAN" and m2 is not None:
                core2 = set(m2.core_sample_indices_)
                db_types2 = np.array(
                    ["core" if i in core2 else ("border" if lbs2[i] != -1 else "noise")
                     for i in range(len(lbs2))])

            fig2 = plot_clustering(X2d, lbs2, centroids=cens2,
                                   feature_names=r["feature_names"],
                                   title=f"{algo2}  ·  {ds_name}",
                                   dbscan_types=db_types2)
            if algo2 == "GMM" and m2 is not None:
                fig2 = add_gmm_ellipses(fig2, m2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Secondary visualization ──
    if res:
        r = res
        show_secondary = False

        # Actual vs Predicted (regression)
        if res["task"] == "Regression":
            y_tr_pred = r["model"].predict(r["X_train"])
            show_secondary = True
            st.markdown('<div class="plot-box" style="margin-top:0.5rem">',
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_actual_vs_predicted(r["y_train"], y_tr_pred, r["y_test"], r["y_pred"]),
                use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Confusion matrix
        if res["task"] == "Classification" and not res.get("is_multilabel", False):
            cm    = sk_metrics.confusion_matrix(r["y_test"], r["y_pred"])
            cn    = r["class_names"] or [str(c) for c in np.unique(r["y_test"])]
            show_secondary = True
            st.markdown('<div class="plot-box" style="margin-top:0.5rem">',
                        unsafe_allow_html=True)
            st.plotly_chart(plot_confusion_matrix(cm, cn),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance
        if hasattr(r.get("model"), "feature_importances_"):
            imps  = r["model"].feature_importances_
            fnames= r.get("feature_names", [f"f{i}" for i in range(len(imps))])
            if len(imps) == len(fnames) and len(imps) <= 30:
                show_secondary = True
                st.markdown('<div class="plot-box" style="margin-top:0.5rem">',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_feature_importance(imps, fnames),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

        # Regression coefficients (linear models)
        if res["task"] == "Regression" and hasattr(r.get("model"), "coef_"):
            coef  = np.atleast_1d(r["model"].coef_)
            fnames= r.get("feature_names", [f"f{i}" for i in range(len(coef))])
            if len(coef) == len(fnames):
                st.markdown('<div class="plot-box" style="margin-top:0.5rem">',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_coefficients(coef, fnames),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

        # Decision tree text
        if algo in ("Decision Tree", "Decision Tree Regressor") and r.get("model"):
            fn = r.get("feature_names", ["f0", "f1"])
            cn = r.get("class_names")
            txt = export_text(r["model"], feature_names=fn,
                              class_names=cn, max_depth=6)
            with st.expander("🌳 Tree Structure"):
                st.code(txt, language="text")

        # Dendrogram
        if algo == "Hierarchical" and r.get("X2d") is not None:
            st.markdown('<div class="plot-box" style="margin-top:0.5rem">',
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_dendrogram_fig(r["X2d"], p.get("linkage", "ward"), p.get("n_clusters", 3)),
                use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    # Step-based secondary: dendrogram placeholder skipped (not needed for K-Means)

    # ── Algorithm description (bottom) ──
    desc = DESCRIPTIONS.get(algo, {})
    with st.expander(f"ℹ️  About: **{algo}**"):
        if desc.get("how"):
            st.markdown(f"**How it works:** {desc['how']}")
        if desc.get("params"):
            st.markdown("**Parameters:**")
            for pname, pdesc in desc["params"].items():
                st.markdown(f"- **{pname}**: {pdesc}")

# ══════════════════════ RIGHT: metrics + info ═════════════════════════════════

with info_col:
    st.markdown('<div class="stats-panel">', unsafe_allow_html=True)
    st.markdown("**📊 Metrics**")

    has_res = res is not None or (steps is not None and st.session_state.step_idx > 0)

    if not has_res:
        st.info("Run the model to see metrics.", icon="💡")

    # Classification metrics
    elif res and res.get("task") == "Classification" and "y_pred" in res:
        r    = res
        is_ml = r.get("is_multilabel", False)
        if is_ml:
            acc  = sk_metrics.accuracy_score(r["y_test"], r["y_pred"])
            st.metric("Subset accuracy", f"{acc:.3f}")
            st.metric("Hamming loss",
                      f"{sk_metrics.hamming_loss(r['y_test'], r['y_pred']):.3f}")
        else:
            y_te, y_pr = r["y_test"], r["y_pred"]
            avg = "binary" if len(np.unique(y_te)) == 2 else "macro"
            cn  = r.get("class_names") or [str(c) for c in np.unique(y_te)]
            st.metric("Accuracy",  f"{sk_metrics.accuracy_score(y_te, y_pr):.3f}")
            st.metric("Precision", f"{sk_metrics.precision_score(y_te, y_pr, average=avg, zero_division=0):.3f}")
            st.metric("Recall",    f"{sk_metrics.recall_score(y_te, y_pr, average=avg, zero_division=0):.3f}")
            st.metric("F1 Score",  f"{sk_metrics.f1_score(y_te, y_pr, average=avg, zero_division=0):.3f}")
            if r.get("y_prob") is not None and len(np.unique(y_te)) == 2:
                auc = sk_metrics.roc_auc_score(y_te, r["y_prob"][:, 1])
                st.metric("ROC AUC", f"{auc:.3f}")
            st.divider()
            with st.expander("📋 Classification Report"):
                report = sk_metrics.classification_report(
                    y_te, y_pr, target_names=cn[:len(np.unique(y_te))], zero_division=0)
                st.code(report, language="text")

    # Regression metrics
    elif res and res.get("task") == "Regression" and "y_pred" in res:
        r    = res
        y_te, y_pr = r["y_test"], r["y_pred"]
        mse  = sk_metrics.mean_squared_error(y_te, y_pr)
        mae  = sk_metrics.mean_absolute_error(y_te, y_pr)
        st.metric("MAE",  f"{mae:.4g}")
        st.metric("MSE",  f"{mse:.4g}")
        st.metric("RMSE", f"{np.sqrt(mse):.4g}")
        st.metric("R²",   f"{sk_metrics.r2_score(y_te, y_pr):.4f}")
        st.metric("Expl. Variance", f"{sk_metrics.explained_variance_score(y_te, y_pr):.4f}")
        try:
            mape = sk_metrics.mean_absolute_percentage_error(y_te, y_pr) * 100
            st.metric("MAPE", f"{mape:.2f}%")
        except Exception:
            pass

    # Clustering metrics
    elif (res and res.get("task") == "Clustering" and "labels" in res) or \
         (steps is not None and st.session_state.step_idx > 0):
        if steps is not None:
            lbs  = steps[st.session_state.step_idx][1]
            X2d  = res["X2d"]
        else:
            lbs  = res["labels"]
            X2d  = res["X2d"]
        n_cls = len(np.unique(lbs[lbs != -1]))
        st.metric("Clusters found", n_cls)
        noise_n = int((lbs == -1).sum())
        if noise_n > 0:
            st.metric("Noise points", noise_n)
        if n_cls > 1 and len(np.unique(lbs)) > 1:
            try:
                sil = sk_metrics.silhouette_score(X2d, lbs)
                st.metric("Silhouette", f"{sil:.3f}")
            except Exception:
                pass
        if hasattr(res.get("model"), "inertia_"):
            st.metric("Inertia", f"{res['model'].inertia_:.2f}")

    # PCA note
    if res and res.get("pca") is not None:
        st.divider()
        st.caption("⚠️ Training on 2 PCA components. Accuracy may be lower than full-feature training.")

    st.markdown('</div>', unsafe_allow_html=True)

# ─── animation loop (must stay last) ─────────────────────────────────────────

if st.session_state.is_running and steps is not None:
    if st.session_state.step_idx < len(steps) - 1:
        st.session_state.step_idx += 1
        time.sleep(delay)
        st.rerun()
    else:
        st.session_state.is_running = False
