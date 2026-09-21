import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from functions import FUNCTIONS
from algorithms import ALGORITHMS

# ─── page config ──────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Optimization Visualizer", page_icon="🎯")

# ─── custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Streamlit top bar: keep it (it holds the sidebar toggle) but make it
       transparent and hide the share/settings buttons inside it ── */
[data-testid="stHeader"]  { background: transparent !important; border-bottom: none !important; box-shadow: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── layout: push content below the (now invisible) header bar ── */
.block-container { padding-top: 3.8rem !important; padding-bottom: 0.5rem !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] > div:first-child {
    background: #F1F5F9;
    padding-top: 0.8rem;
}
[data-testid="stSidebar"] hr { border-color: #CBD5E1 !important; margin: 0.6rem 0 !important; }
[data-testid="stSidebar"] h3 { color: #1E3A5F !important; font-size: 0.85rem !important;
                                 text-transform: uppercase; letter-spacing: 0.06em;
                                 margin: 0.4rem 0 0.1rem 0 !important; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.4rem 0.7rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 5px;
}
[data-testid="stMetricValue"]  { color: #1E293B !important; font-size: 1.05rem !important; }
[data-testid="stMetricLabel"]  { color: #64748B !important; font-size: 0.72rem !important; }

/* ── control buttons ── */
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    border: 1px solid #CBD5E1 !important;
    transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* ── stats box ── */
.stats-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    height: 100%;
}
.stats-title { font-size: 0.95rem; font-weight: 700; color: #1E293B;
               margin-bottom: 0.6rem; display: flex; align-items: center; gap: 6px; }

/* ── plot wrapper ── */
.plot-box {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ─── session state ────────────────────────────────────────────────────────────

for k, v in {"opt_state": None, "is_running": False, "run_key": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── sidebar ──────────────────────────────────────────────────────────────────

SINGLE_AGENT = {"Random Walk", "Monte Carlo", "Metropolis Monte Carlo", "Simulated Annealing"}

with st.sidebar:
    st.markdown("## 🎯 Visualizer")

    st.divider()
    st.subheader("Algorithm & Function")

    algo_name = st.selectbox("Algorithm", list(ALGORITHMS.keys()), label_visibility="collapsed")
    algo = ALGORITHMS[algo_name]

    dims = st.radio("Dimensions", [1, 2], horizontal=True)

    fn_choices = [n for n, f in FUNCTIONS.items() if dims in f.dims]
    fn_name = st.selectbox("Function", fn_choices, label_visibility="collapsed")
    fn_obj = FUNCTIONS[fn_name]

    st.divider()
    st.subheader("Search Space")

    bc1, bc2 = st.columns(2)
    b_min = bc1.number_input("Min", value=float(fn_obj.bounds[0]), step=0.5, format="%.1f")
    b_max = bc2.number_input("Max", value=float(fn_obj.bounds[1]), step=0.5, format="%.1f")

    st.divider()
    st.subheader("Run Settings")

    n_iters = st.slider("Iterations", 20, 1000, 150)
    seed = int(st.number_input("Random Seed", value=42, min_value=0, step=1))

    st.divider()
    st.subheader("Parameters")

    extra = {}

    if algo_name == "Random Walk":
        extra["step_size"] = st.slider("Move distance (fraction of range)", 0.01, 0.40, 0.05, 0.01)

    elif algo_name == "Monte Carlo":
        st.caption("Pure random search — no extra parameters.")

    elif algo_name == "Metropolis Monte Carlo":
        extra["temperature"] = st.slider("Temperature  T", 0.01, 20.0, 1.0, 0.05)
        extra["step_size"]   = st.slider("Step size (fraction of range)", 0.01, 0.40, 0.05, 0.01)

    elif algo_name == "Simulated Annealing":
        extra["initial_temp"] = st.slider("Initial temperature  T₀", 0.5, 100.0, 10.0, 0.5)
        extra["cooling_rate"] = st.slider("Cooling rate  α  (T = T₀·αⁿ)", 0.800, 0.999, 0.950, 0.001, format="%.3f")
        extra["step_size"]    = st.slider("Step size (fraction of range)", 0.01, 0.40, 0.10, 0.01)

    elif algo_name == "Genetic Algorithm":
        extra["population_size"] = st.slider("Population size", 5, 80, 20)
        extra["crossover_rate"]  = st.slider("Crossover probability", 0.50, 1.00, 0.80, 0.05)
        extra["mutation_rate"]   = st.slider("Mutation probability", 0.01, 0.50, 0.15, 0.01)

    elif algo_name == "Ant Colony Optimization":
        colony = st.slider("Colony size", 5, 60, 15)
        extra["n_ants"] = colony
        extra["archive_size"] = colony
        q_raw = st.slider("Exploitation ◀──────▶ Exploration", 0.0, 1.0, 0.3, 0.05)
        extra["q"] = 0.1 + q_raw * 1.9   # map [0,1] → [0.1, 2.0]
        extra["xi"] = 0.85

    elif algo_name == "Black Hole":
        extra["population_size"] = st.slider("Population size", 5, 80, 20)

# ─── config & auto-reset ──────────────────────────────────────────────────────

if b_min >= b_max:
    st.error("Bounds min must be less than max.")
    st.stop()

config = {"dims": dims, "bounds": (b_min, b_max), "seed": seed, **extra}

run_key = f"{algo_name}|{fn_name}|{dims}"
if st.session_state.run_key != run_key:
    st.session_state.opt_state = None
    st.session_state.is_running = False
    st.session_state.run_key = run_key

# ─── header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(135deg,#1E40AF 0%,#3B82F6 100%);
            padding:1rem 1.4rem;border-radius:12px;margin-bottom:0.8rem;">
  <span style="color:white;font-size:1.5rem;font-weight:700;">🎯 Optimization Algorithm Visualizer</span><br>
  <span style="color:#BFDBFE;font-size:0.85rem;">
    Watch algorithms search for the global minimum — step by step or in real time
  </span>
</div>
""", unsafe_allow_html=True)

# ─── control row ──────────────────────────────────────────────────────────────

b1, b2, b3, b4, sp, spd_col = st.columns([1, 1, 1, 1, 0.3, 3.5])
run_btn   = b1.button("▶  Run",   type="primary", use_container_width=True)
pause_btn = b2.button("⏸  Pause",                 use_container_width=True)
step_btn  = b3.button("⏭  Step",                  use_container_width=True)
reset_btn = b4.button("↺  Reset",                 use_container_width=True)

with spd_col:
    scA, scB, scC = st.columns([1, 6, 1])
    scA.markdown("<div style='text-align:right;font-size:0.78rem;color:#64748B;padding-top:8px'>🐢</div>",
                 unsafe_allow_html=True)
    speed_val = scB.slider("speed", 1, 10, 6, label_visibility="collapsed")
    scC.markdown("<div style='font-size:0.78rem;color:#64748B;padding-top:8px'>🚀</div>",
                 unsafe_allow_html=True)

# delay curve: 0.8 s (speed=1) → 0.01 s (speed=10)
delay = round(0.8 * (0.01 / 0.8) ** ((speed_val - 1) / 9), 3)

# Handle buttons
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
            st.session_state.opt_state, config, fn_obj.fn)

state = st.session_state.opt_state

# ─── iteration progress bar ───────────────────────────────────────────────────

if state:
    progress = min(state["iteration"] / n_iters, 1.0)
    status   = "🟢 Running" if st.session_state.is_running else \
               ("✅ Done"   if state["iteration"] >= n_iters else "⏸ Paused")
    st.progress(progress, text=f"{status} — iteration {state['iteration']} / {n_iters}")
else:
    st.progress(0.0, text="Press ▶ Run or ⏭ Step to start")

# ─── cached function grids ────────────────────────────────────────────────────

@st.cache_data
def _curve_1d(fn_name, b_min, b_max):
    fn = FUNCTIONS[fn_name].fn
    xs = np.linspace(b_min, b_max, 500)
    return xs, np.array([fn(np.array([x])) for x in xs])


@st.cache_data
def _grid_2d(fn_name, b_min, b_max, n=70):
    fn = FUNCTIONS[fn_name].fn
    xs = np.linspace(b_min, b_max, n)
    ys = np.linspace(b_min, b_max, n)
    Z  = np.array([[fn(np.array([x, y])) for x in xs] for y in ys])
    return xs, ys, Z

# ─── plot helpers ─────────────────────────────────────────────────────────────

_LAYOUT = dict(
    template="simple_white",
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0",
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=12, r=12, t=44, b=12),
)

C_CURVE  = "#2563EB"   # blue  – function line
C_AGENT  = "#DC2626"   # red   – current candidates
C_BEST   = "#D97706"   # amber – best found
C_CONV   = "#059669"   # green – convergence


def _plot_1d(fn_obj, config, state):
    xs, ys = _curve_1d(fn_obj.name, config["bounds"][0], config["bounds"][1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=fn_obj.name,
                             line=dict(color=C_CURVE, width=2.5)))
    if state is not None:
        cx, cy = state["candidates"][:, 0], state["values"]
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", name="Current",
                                 marker=dict(color=C_AGENT, size=13, symbol="circle",
                                             line=dict(color="white", width=1.5))))
        fig.add_trace(go.Scatter(
            x=[state["best_pos"][0]], y=[state["best_val"]],
            mode="markers", name="Best",
            marker=dict(color=C_BEST, size=19, symbol="star",
                        line=dict(color="#78350F", width=1.5)),
        ))
    fig.update_layout(**_LAYOUT, height=420,
                      title=dict(text=f"<b>{fn_obj.name}</b> — 1D landscape",
                                 font=dict(size=15, color="#1E293B")),
                      xaxis=dict(title="x", showgrid=True, gridcolor="#F1F5F9",
                                 zeroline=True, zerolinecolor="#CBD5E1"),
                      yaxis=dict(title="f(x)", showgrid=True, gridcolor="#F1F5F9"))
    return fig


def _plot_2d(fn_obj, config, state):
    b_min, b_max = config["bounds"]
    xs, ys, Z = _grid_2d(fn_obj.name, b_min, b_max)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xs, y=ys, z=Z, colorscale="RdYlBu_r",
                             contours=dict(coloring="heatmap", showlabels=False),
                             showscale=True, colorbar=dict(thickness=12, len=0.85),
                             name=fn_obj.name))
    if state is not None:
        cx, cy = state["candidates"][:, 0], state["candidates"][:, 1]
        # draw many agents or just 1 depending on algorithm type
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers",
                                 name="Agent" if algo_name in SINGLE_AGENT else "Population",
                                 marker=dict(color="white", size=10, symbol="circle",
                                             line=dict(color="#1E293B", width=1.5))))
        fig.add_trace(go.Scatter(
            x=[state["best_pos"][0]], y=[state["best_pos"][1]],
            mode="markers", name="Best",
            marker=dict(color=C_BEST, size=19, symbol="star",
                        line=dict(color="#78350F", width=1.5)),
        ))
    fig.update_layout(**_LAYOUT, height=460,
                      title=dict(text=f"<b>{fn_obj.name}</b> — 2D landscape",
                                 font=dict(size=15, color="#1E293B")),
                      xaxis=dict(title="x₁", range=[b_min, b_max], showgrid=False),
                      yaxis=dict(title="x₂", range=[b_min, b_max], showgrid=False))
    return fig


def _conv_plot(state):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=state["history"], mode="lines", name="Best value",
                             line=dict(color=C_CONV, width=2),
                             fill="tozeroy", fillcolor="rgba(5,150,105,0.08)"))
    fig.update_layout(**_LAYOUT, height=170,
                      title=dict(text="<b>Convergence</b> — best value vs iteration",
                                 font=dict(size=13, color="#1E293B")),
                      xaxis=dict(title="Iteration", showgrid=True, gridcolor="#F1F5F9"),
                      yaxis=dict(title="f(best)", showgrid=True, gridcolor="#F1F5F9"),
                      showlegend=False)
    return fig

# ─── main plot + stats ────────────────────────────────────────────────────────

plot_col, stats_col = st.columns([3, 1])

with plot_col:
    st.markdown('<div class="plot-box">', unsafe_allow_html=True)
    fig = _plot_1d(fn_obj, config, state) if dims == 1 else _plot_2d(fn_obj, config, state)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with stats_col:
    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
    st.markdown('<div class="stats-title">📊 Statistics</div>', unsafe_allow_html=True)

    if state is None:
        st.info("Press **▶ Run** or **⏭ Step** to start.", icon="💡")
    else:
        st.metric("Best value", f"{state['best_val']:.6g}")

        pos = state["best_pos"]
        if dims == 1:
            st.metric("Best  x", f"{pos[0]:.5g}")
        else:
            st.metric("Best  x₁", f"{pos[0]:.5g}")
            st.metric("Best  x₂", f"{pos[1]:.5g}")

        st.metric("Fn evaluations", f"{state['n_evals']:,}")

        if "temperature" in state:
            st.metric("Temperature  T", f"{state['temperature']:.4g}")

        # Population stats for multi-agent algorithms
        if len(state["values"]) > 1:
            st.divider()
            st.caption("**Population stats**")
            v = state["values"]
            ca, cb = st.columns(2)
            ca.metric("Mean", f"{np.mean(v):.4g}")
            cb.metric("Std",  f"{np.std(v):.4g}")
            ca.metric("Min",  f"{np.min(v):.4g}")
            cb.metric("Max",  f"{np.max(v):.4g}")

    st.markdown('</div>', unsafe_allow_html=True)

# ─── convergence graph ────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:0.5rem'>", unsafe_allow_html=True)
if state and len(state["history"]) > 1:
    st.markdown('<div class="plot-box">', unsafe_allow_html=True)
    st.plotly_chart(_conv_plot(state), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.caption("Convergence chart appears after the first step.")
st.markdown("</div>", unsafe_allow_html=True)

# ─── algorithm description ────────────────────────────────────────────────────

with st.expander(f"ℹ️  About: **{algo_name}**  ·  Function: **{fn_name}**"):
    st.markdown(f"**{algo_name}:** {algo.description}")
    st.markdown(f"**{fn_name}:** {fn_obj.description}")

# ─── animation loop (must stay last) ─────────────────────────────────────────

if st.session_state.is_running:
    if state is None or state["iteration"] >= n_iters:
        st.session_state.is_running = False
    else:
        st.session_state.opt_state = algo.step(state, config, fn_obj.fn)
        time.sleep(delay)
        st.rerun()
