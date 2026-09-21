import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from functions import FUNCTIONS
from algorithms import ALGORITHMS

# ─── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Optimization Visualizer",
    page_icon="🎯",
)

# ─── session state defaults ───────────────────────────────────────────────────

for k, v in {"opt_state": None, "is_running": False, "run_key": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Controls")

    algo_name = st.selectbox("Algorithm", list(ALGORITHMS.keys()))
    algo = ALGORITHMS[algo_name]

    dims = st.radio("Dimensions", [1, 2], horizontal=True)

    fn_choices = [name for name, f in FUNCTIONS.items() if dims in f.dims]
    fn_name = st.selectbox("Function", fn_choices)
    fn_obj = FUNCTIONS[fn_name]

    st.divider()
    st.subheader("Search Space & Run")

    col_a, col_b = st.columns(2)
    b_min = col_a.number_input("Min", value=float(fn_obj.bounds[0]), step=0.5, format="%.1f")
    b_max = col_b.number_input("Max", value=float(fn_obj.bounds[1]), step=0.5, format="%.1f")

    n_iters = st.slider("Max Iterations", 10, 500, 100)
    seed = int(st.number_input("Random Seed", value=42, min_value=0, step=1))

    speed_label = st.select_slider(
        "Speed", ["Slowest", "Slow", "Normal", "Fast", "Fastest"], value="Normal"
    )
    delay = {"Slowest": 0.8, "Slow": 0.35, "Normal": 0.12, "Fast": 0.04, "Fastest": 0.005}[speed_label]

    st.divider()
    st.subheader(f"{algo_name} params")

    extra = {}
    if algo_name == "Random Walk":
        extra["step_size"] = st.slider("Step Size (fraction)", 0.01, 0.50, 0.05, 0.01)
    elif algo_name == "Monte Carlo":
        extra["n_samples"] = st.slider("Samples / iteration", 5, 100, 20)
    elif algo_name == "Metropolis Monte Carlo":
        extra["temperature"] = st.slider("Temperature T", 0.01, 20.0, 1.0, 0.05)
        extra["step_size"]   = st.slider("Step Size (fraction)", 0.01, 0.50, 0.05, 0.01)
    elif algo_name == "Simulated Annealing":
        extra["initial_temp"]  = st.slider("Initial Temp T₀", 0.5, 100.0, 10.0, 0.5)
        extra["cooling_rate"]  = st.slider("Cooling Rate α", 0.800, 0.999, 0.950, 0.001, format="%.3f")
        extra["step_size"]     = st.slider("Step Size (fraction)", 0.01, 0.50, 0.10, 0.01)
    elif algo_name == "Genetic Algorithm":
        extra["population_size"] = st.slider("Population Size", 5, 100, 20)
        extra["mutation_rate"]   = st.slider("Mutation Rate", 0.01, 0.50, 0.15, 0.01)
        extra["crossover_rate"]  = st.slider("Crossover Rate", 0.50, 1.00, 0.80, 0.05)
    elif algo_name == "Ant Colony Optimization":
        extra["n_ants"]       = st.slider("# Ants / iteration", 5, 50, 10)
        extra["archive_size"] = st.slider("Archive Size", 5, 50, 10)
        extra["xi"]           = st.slider("ξ (locality)", 0.1, 2.0, 0.85, 0.05)
    elif algo_name == "Black Hole":
        extra["population_size"] = st.slider("Population Size", 5, 100, 20)

# ─── build config ─────────────────────────────────────────────────────────────

if b_min >= b_max:
    st.error("Bounds min must be less than max.")
    st.stop()

config = {"dims": dims, "bounds": (b_min, b_max), "seed": seed, **extra}

# Reset state automatically when algorithm / function / dimensionality changes
run_key = f"{algo_name}|{fn_name}|{dims}"
if st.session_state.run_key != run_key:
    st.session_state.opt_state = None
    st.session_state.is_running = False
    st.session_state.run_key = run_key

# ─── title & control buttons ──────────────────────────────────────────────────

st.title("🎯 Optimization Algorithm Visualizer")

btn_cols = st.columns([1, 1, 1, 1, 3])
run_btn   = btn_cols[0].button("▶ Run",    type="primary", use_container_width=True)
pause_btn = btn_cols[1].button("⏸ Pause",                  use_container_width=True)
step_btn  = btn_cols[2].button("⏭ Step",                   use_container_width=True)
reset_btn = btn_cols[3].button("↺ Reset",                  use_container_width=True)

# Handle button clicks
if reset_btn:
    st.session_state.opt_state = None
    st.session_state.is_running = False

if pause_btn:
    st.session_state.is_running = False

if run_btn:
    st.session_state.is_running = True
    if st.session_state.opt_state is None:
        st.session_state.opt_state = algo.initialize(config, fn_obj.fn)

if step_btn:
    st.session_state.is_running = False
    if st.session_state.opt_state is None:
        st.session_state.opt_state = algo.initialize(config, fn_obj.fn)
    elif st.session_state.opt_state["iteration"] < n_iters:
        st.session_state.opt_state = algo.step(
            st.session_state.opt_state, config, fn_obj.fn
        )

state = st.session_state.opt_state

# ─── cached function grids ────────────────────────────────────────────────────

@st.cache_data
def _fn_curve_1d(fn_name, b_min, b_max):
    fn = FUNCTIONS[fn_name].fn
    xs = np.linspace(b_min, b_max, 500)
    ys = np.array([fn(np.array([x])) for x in xs])
    return xs, ys


@st.cache_data
def _fn_grid_2d(fn_name, b_min, b_max, n=60):
    fn = FUNCTIONS[fn_name].fn
    xs = np.linspace(b_min, b_max, n)
    ys = np.linspace(b_min, b_max, n)
    Z = np.array([[fn(np.array([x, y])) for x in xs] for y in ys])
    return xs, ys, Z


# ─── plot helpers ─────────────────────────────────────────────────────────────

DARK = "plotly_dark"
GOLD = "#FFD700"
ORANGE = "#FF9F43"
BLUE = "#4DABF7"


def _plot_1d(fn_obj, config, state):
    xs, ys = _fn_curve_1d(fn_obj.name, config["bounds"][0], config["bounds"][1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=fn_obj.name,
                             line=dict(color=BLUE, width=2.5)))
    if state is not None:
        cx = state["candidates"][:, 0]
        cy = state["values"]
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", name="Candidate(s)",
                                 marker=dict(color=ORANGE, size=12, symbol="circle",
                                             line=dict(color="white", width=1))))
        fig.add_trace(go.Scatter(
            x=[state["best_pos"][0]], y=[state["best_val"]],
            mode="markers", name="Best",
            marker=dict(color=GOLD, size=18, symbol="star",
                        line=dict(color="black", width=1)),
        ))
    fig.update_layout(
        template=DARK, height=440,
        title=dict(text=f"{fn_obj.name} — 1D", font=dict(size=16)),
        xaxis_title="x", yaxis_title="f(x)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def _plot_2d(fn_obj, config, state):
    b_min, b_max = config["bounds"]
    xs, ys, Z = _fn_grid_2d(fn_obj.name, b_min, b_max)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xs, y=ys, z=Z, colorscale="Viridis",
                             contours=dict(coloring="heatmap", showlabels=False),
                             showscale=True, name=fn_obj.name))
    if state is not None:
        cx = state["candidates"][:, 0]
        cy = state["candidates"][:, 1]
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", name="Agents",
                                 marker=dict(color="white", size=9, symbol="circle",
                                             line=dict(color="black", width=1))))
        fig.add_trace(go.Scatter(
            x=[state["best_pos"][0]], y=[state["best_pos"][1]],
            mode="markers", name="Best",
            marker=dict(color=GOLD, size=18, symbol="star",
                        line=dict(color="black", width=1)),
        ))
    fig.update_layout(
        template=DARK, height=480,
        title=dict(text=f"{fn_obj.name} — 2D", font=dict(size=16)),
        xaxis=dict(title="x₁", range=[b_min, b_max]),
        yaxis=dict(title="x₂", range=[b_min, b_max]),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def _convergence_plot(state):
    history = state["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history, mode="lines", name="Best value",
                             line=dict(color=GOLD, width=2),
                             fill="tozeroy", fillcolor="rgba(255,215,0,0.08)"))
    fig.update_layout(
        template=DARK, height=180,
        title=dict(text="Convergence — best value vs iteration", font=dict(size=14)),
        xaxis_title="Iteration", yaxis_title="f(best)",
        margin=dict(l=10, r=10, t=38, b=10),
        showlegend=False,
    )
    return fig


# ─── main layout ──────────────────────────────────────────────────────────────

plot_col, stats_col = st.columns([3, 1])

with plot_col:
    if dims == 1:
        st.plotly_chart(_plot_1d(fn_obj, config, state), use_container_width=True)
    else:
        st.plotly_chart(_plot_2d(fn_obj, config, state), use_container_width=True)

with stats_col:
    st.subheader("📊 Statistics")
    if state is None:
        st.info("Press **Run** or **Step** to start.")
    else:
        it = state["iteration"]
        status = "Running 🟢" if st.session_state.is_running else ("Done ✅" if it >= n_iters else "Paused ⏸")
        st.caption(status)
        st.metric("Iteration", f"{it} / {n_iters}")
        st.metric("Best value", f"{state['best_val']:.6g}")

        pos = state["best_pos"]
        if len(pos) == 1:
            st.metric("Best x", f"{pos[0]:.5g}")
        else:
            st.metric("Best x₁", f"{pos[0]:.5g}")
            st.metric("Best x₂", f"{pos[1]:.5g}")

        st.metric("Fn evaluations", state["n_evals"])

        if "temperature" in state:
            st.metric("Temperature", f"{state['temperature']:.4g}")

        vals = state["values"]
        if len(vals) > 1:
            st.divider()
            st.caption("Population")
            c1, c2 = st.columns(2)
            c1.metric("Mean", f"{np.mean(vals):.4g}")
            c2.metric("Std",  f"{np.std(vals):.4g}")
            c1.metric("Min",  f"{np.min(vals):.4g}")
            c2.metric("Max",  f"{np.max(vals):.4g}")

# ─── convergence graph ────────────────────────────────────────────────────────

if state and len(state["history"]) > 1:
    st.plotly_chart(_convergence_plot(state), use_container_width=True)
else:
    st.caption("Convergence graph will appear after the first step.")

# ─── algorithm description ────────────────────────────────────────────────────

with st.expander(f"ℹ️ About: {algo_name}", expanded=False):
    st.write(algo.description)
    st.caption(f"**Function:** {fn_obj.description}")

# ─── animation loop (must be last) ───────────────────────────────────────────

if st.session_state.is_running:
    if state is None or state["iteration"] >= n_iters:
        st.session_state.is_running = False
    else:
        st.session_state.opt_state = algo.step(state, config, fn_obj.fn)
        time.sleep(delay)
        st.rerun()
