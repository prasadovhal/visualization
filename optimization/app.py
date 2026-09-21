import ast
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from functions import FUNCTIONS, ObjFn
from algorithms import ALGORITHMS

# ─── custom function validator ────────────────────────────────────────────────

_SAFE_NS = {
    "__builtins__": {},
    "np": np, "math": math,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e,
    "inf": np.inf, "nan": np.nan,
}
_ALLOWED_NAMES = set(_SAFE_NS.keys()) | {"x", "y"}


def _validate_custom(expr, dims):
    """Return (callable, error_str). callable is None on failure."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as err:
        return None, f"Syntax error: {err.msg}"

    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    bad = names - _ALLOWED_NAMES
    if bad:
        return None, f"Unknown names: {sorted(bad)}. Use x, y, np.sin/cos/exp/log/sqrt/…, pi, e."
    if dims == 1 and "y" in names:
        return None, "1D selected — only 'x' is allowed. Remove 'y'."
    if dims == 2 and names <= {"y"} and "x" not in names:
        return None, "2D selected — expression must use at least 'x'."

    if dims == 1:
        def fn(pos):
            try:
                v = float(eval(expr, _SAFE_NS, {"x": float(pos[0])}))
                return v if np.isfinite(v) else 1e10
            except Exception:
                return 1e10
    else:
        def fn(pos):
            try:
                v = float(eval(expr, _SAFE_NS, {"x": float(pos[0]), "y": float(pos[1])}))
                return v if np.isfinite(v) else 1e10
            except Exception:
                return 1e10

    # smoke-test at origin
    try:
        result = fn(np.zeros(dims))
        if result == 1e10:
            return None, "Function returned non-finite value at x=0. Check expression."
    except Exception as err:
        return None, f"Evaluation failed: {err}"

    return fn, None

# ─── page config ──────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Optimization Visualizer", page_icon="🎯")

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* global */
.block-container { padding: 0.8rem 1.2rem 0.5rem 1.2rem !important; }
[data-testid="stHeader"]     { background: transparent !important; border: none !important; box-shadow: none !important; }
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* control panel card */
.ctrl-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 0.9rem;
}
.ctrl-section {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94A3B8;
    margin: 0.9rem 0 0.3rem 0;
}

/* stats card */
.stats-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.9rem;
}

/* metric cards */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.3rem 0.6rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 4px;
}
[data-testid="stMetricValue"] { color: #1E293B !important; font-size: 1rem !important; }
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.7rem !important; }

/* buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid #CBD5E1 !important;
}
.stButton > button:hover { filter: brightness(0.95) !important; }

/* plot box */
.plot-box {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ─── session state ────────────────────────────────────────────────────────────

for k, v in {"opt_state": None, "all_steps": None, "run_key": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(135deg,#1E40AF 0%,#3B82F6 100%);
            padding:0.85rem 1.3rem;border-radius:12px;margin-bottom:0.7rem;">
  <span style="color:white;font-size:1.4rem;font-weight:700;">🎯 Optimization Algorithm Visualizer</span><br>
  <span style="color:#BFDBFE;font-size:0.82rem;">
    Watch algorithms search for the global minimum — step by step or in real time
  </span>
</div>
""", unsafe_allow_html=True)

# ─── 3-column layout ──────────────────────────────────────────────────────────

SINGLE_AGENT = {"Random Walk", "Monte Carlo", "Metropolis Monte Carlo", "Simulated Annealing"}

ctrl_col, plot_col, stats_col = st.columns([1.15, 2.9, 1.15])

# ══════════════════════════ LEFT: controls ════════════════════════════════════

with ctrl_col:
    st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)

    st.markdown('<div class="ctrl-section">Algorithm & Function</div>', unsafe_allow_html=True)
    algo_name = st.selectbox("Algorithm", list(ALGORITHMS.keys()), label_visibility="collapsed")
    algo = ALGORITHMS[algo_name]

    dims = st.radio("Dimensions", [1, 2], horizontal=True)

    fn_choices = [n for n, f in FUNCTIONS.items() if dims in f.dims] + ["Custom..."]
    fn_name = st.selectbox("Function", fn_choices, label_visibility="collapsed")

    # ── custom function input ──
    custom_expr = ""
    custom_fn   = None
    if fn_name == "Custom...":
        _ph = "x**2 + np.sin(3*x)" if dims == 1 else "x**2 + y**2"
        _lbl = "f(x) =" if dims == 1 else "f(x, y) ="
        custom_expr = st.text_input(_lbl, placeholder=_ph, key=f"cexpr_{dims}",
                                    label_visibility="visible")
        if custom_expr.strip():
            custom_fn, _cerr = _validate_custom(custom_expr.strip(), dims)
            if _cerr:
                st.error(_cerr, icon="🚫")
            else:
                st.success("Valid expression", icon="✅")
        else:
            st.caption("Enter expression using `x`" + (" or `x`, `y`" if dims == 2 else "") + ".")

    _def_bounds = FUNCTIONS[fn_name].bounds if fn_name != "Custom..." else (-5.0, 5.0)
    fn_obj = FUNCTIONS[fn_name] if fn_name != "Custom..." else ObjFn(
        "Custom", custom_fn, [dims], _def_bounds, custom_expr or "User-defined function"
    )

    st.markdown('<div class="ctrl-section">Search Space</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    b_min = bc1.number_input("Min", value=float(_def_bounds[0]), step=0.5, format="%.1f")
    b_max = bc2.number_input("Max", value=float(_def_bounds[1]), step=0.5, format="%.1f")

    st.markdown('<div class="ctrl-section">Run Settings</div>', unsafe_allow_html=True)
    n_iters = st.slider("Iterations", 20, 1000, 150)
    seed    = int(st.number_input("Seed", value=42, min_value=0, step=1))

    st.markdown('<div class="ctrl-section">Parameters</div>', unsafe_allow_html=True)
    extra = {}

    if algo_name == "Random Walk":
        extra["step_size"] = st.slider("Move distance (fraction)", 0.01, 0.40, 0.05, 0.01)

    elif algo_name == "Monte Carlo":
        st.caption("Pure random search — no parameters.")

    elif algo_name == "Metropolis Monte Carlo":
        extra["temperature"] = st.slider("Temperature  T", 0.01, 20.0, 1.0, 0.05)
        extra["step_size"]   = st.slider("Step size (fraction)", 0.01, 0.40, 0.05, 0.01)

    elif algo_name == "Simulated Annealing":
        extra["initial_temp"] = st.slider("Initial temp  T₀", 0.5, 100.0, 10.0, 0.5)
        extra["cooling_rate"] = st.slider("Cooling rate  α", 0.800, 0.999, 0.950, 0.001, format="%.3f")
        extra["step_size"]    = st.slider("Step size (fraction)", 0.01, 0.40, 0.10, 0.01)

    elif algo_name == "Genetic Algorithm":
        extra["population_size"] = st.slider("Population", 5, 80, 20)
        extra["crossover_rate"]  = st.slider("Crossover prob.", 0.50, 1.00, 0.80, 0.05)
        extra["mutation_rate"]   = st.slider("Mutation prob.", 0.01, 0.50, 0.15, 0.01)

    elif algo_name == "Ant Colony Optimization":
        colony       = st.slider("Colony size", 5, 60, 15)
        extra["n_ants"]       = colony
        extra["archive_size"] = colony
        q_raw = st.slider("Exploitation ◀──▶ Exploration", 0.0, 1.0, 0.3, 0.05)
        extra["q"]  = 0.1 + q_raw * 1.9
        extra["xi"] = 0.85

    elif algo_name == "Black Hole":
        extra["population_size"] = st.slider("Population", 5, 80, 20)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── config & auto-reset ──────────────────────────────────────────────────────

if b_min >= b_max:
    with plot_col:
        st.error("Bounds min must be less than max.")
    st.stop()

if fn_name == "Custom..." and custom_fn is None:
    with plot_col:
        st.info("Enter a valid function expression in the left panel.", icon="✏️")
    st.stop()

config  = {"dims": dims, "bounds": (b_min, b_max), "seed": seed, **extra}
run_key = f"{algo_name}|{fn_name}|{dims}|{custom_expr}"
if st.session_state.run_key != run_key:
    st.session_state.opt_state = None
    st.session_state.all_steps = None
    st.session_state.run_key   = run_key

# ══════════════════════════ CENTRE: plot ══════════════════════════════════════

with plot_col:

    # — control buttons + speed —
    b1, b2, b3, _sp, spd = st.columns([1, 1, 1, 0.2, 3.2])
    run_btn   = b1.button("▶ Run",   type="primary", use_container_width=True)
    step_btn  = b2.button("⏭ Step",                  use_container_width=True)
    reset_btn = b3.button("↺ Reset",                 use_container_width=True)

    with spd:
        sa, sb, sc = st.columns([1, 5, 1])
        sa.markdown("<p style='text-align:right;margin-top:8px;font-size:0.8rem'>🐢</p>",
                    unsafe_allow_html=True)
        speed_val = sb.slider("spd", 1, 10, 6, label_visibility="collapsed")
        sc.markdown("<p style='margin-top:8px;font-size:0.8rem'>🚀</p>",
                    unsafe_allow_html=True)

    # speed → Plotly frame duration (ms)
    _raw_delay = round(0.8 * (0.01 / 0.8) ** ((speed_val - 1) / 9), 3)
    frame_ms   = max(30, int(_raw_delay * 1000))

    # — button logic —
    if reset_btn:
        st.session_state.opt_state = None
        st.session_state.all_steps = None

    if step_btn:
        st.session_state.all_steps = None          # exit animated mode
        if st.session_state.opt_state is None:
            st.session_state.opt_state = algo.initialize(config, fn_obj.fn)
        elif st.session_state.opt_state["iteration"] < n_iters:
            st.session_state.opt_state = algo.step(
                st.session_state.opt_state, config, fn_obj.fn)

    if run_btn:
        # Compute ALL remaining steps up-front; Plotly plays them client-side
        if st.session_state.opt_state is None:
            st.session_state.opt_state = algo.initialize(config, fn_obj.fn)
        cur = st.session_state.opt_state
        steps = [cur]
        while cur["iteration"] < n_iters:
            cur = algo.step(cur, config, fn_obj.fn)
            steps.append(cur)
        st.session_state.opt_state = cur
        st.session_state.all_steps = steps

    state     = st.session_state.opt_state
    all_steps = st.session_state.all_steps

    # — progress bar —
    if state:
        pct    = min(state["iteration"] / n_iters, 1.0)
        status = "✅ Done" if all_steps is not None else \
                 ("✅ Done" if state["iteration"] >= n_iters else "⏸ Paused")
        st.progress(pct, text=f"{status}  —  iteration {state['iteration']} / {n_iters}")
    else:
        st.progress(0.0, text="Press ▶ Run or ⏭ Step to start")

    # — cached grids (built-in functions) —
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

    def _get_curve_1d():
        if fn_name == "Custom...":
            xs = np.linspace(b_min, b_max, 500)
            return xs, np.array([fn_obj.fn(np.array([x])) for x in xs])
        return _curve_1d(fn_name, b_min, b_max)

    def _get_grid_2d(n=70):
        if fn_name == "Custom...":
            xs = np.linspace(b_min, b_max, n)
            ys = np.linspace(b_min, b_max, n)
            Z  = np.array([[fn_obj.fn(np.array([x, y])) for x in xs] for y in ys])
            return xs, ys, Z
        return _grid_2d(fn_name, b_min, b_max)

    # — shared styling —
    _L = dict(
        template="simple_white",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#374151", size=12),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=12, r=12, t=44, b=12),
    )
    C_CURVE, C_AGENT, C_BEST, C_CONV = "#2563EB", "#DC2626", "#D97706", "#059669"

    # — stable curve / y-range for 1D —
    _xs1d, _ys1d = _get_curve_1d()
    _y_pad = max(abs(_ys1d.max() - _ys1d.min()) * 0.12, 0.1)
    _y_lo  = float(_ys1d.min() - _y_pad)
    _y_hi  = float(_ys1d.max() + _y_pad)

    # ── helpers that build the two dynamic traces for a single state ──
    def _agent_trace_1d(s):
        cx, cy = s["candidates"][:, 0].tolist(), s["values"].tolist()
        if algo_name == "Ant Colony Optimization":
            return go.Scatter(x=cx, y=cy, mode="text",
                              text=["🐜"] * len(cx), textfont=dict(size=16), name="Ants")
        return go.Scatter(x=cx, y=cy, mode="markers", name="Current",
                          marker=dict(color=C_AGENT, size=13, symbol="circle",
                                      line=dict(color="white", width=1.5)))

    def _best_trace_1d(s):
        return go.Scatter(x=[float(s["best_pos"][0])], y=[float(s["best_val"])],
                          mode="markers", name="Best",
                          marker=dict(color=C_BEST, size=19, symbol="star",
                                      line=dict(color="#78350F", width=1.5)))

    def _agent_trace_2d(s):
        cx = s["candidates"][:, 0].tolist(); cy = s["candidates"][:, 1].tolist()
        if algo_name == "Ant Colony Optimization":
            return go.Scatter(x=cx, y=cy, mode="text",
                              text=["🐜"] * len(cx), textfont=dict(size=18), name="Ants")
        label = "Agent" if algo_name in SINGLE_AGENT else "Population"
        return go.Scatter(x=cx, y=cy, mode="markers", name=label,
                          marker=dict(color="white", size=10, symbol="circle",
                                      line=dict(color="#1E293B", width=1.5)))

    def _best_trace_2d(s):
        return go.Scatter(x=[float(s["best_pos"][0])], y=[float(s["best_pos"][1])],
                          mode="markers", name="Best",
                          marker=dict(color=C_BEST, size=19, symbol="star",
                                      line=dict(color="#78350F", width=1.5)))

    # ── animation config dicts ──
    def _anim_cfg():
        return dict(frame=dict(duration=frame_ms, redraw=False),
                    fromcurrent=True,
                    transition=dict(duration=int(frame_ms * 0.5), easing="linear"))

    def _imm_cfg():
        return dict(frame=dict(duration=0, redraw=False),
                    mode="immediate", transition=dict(duration=0))

    def _anim_menus():
        return [dict(
            type="buttons", showactive=False, direction="left",
            x=0.0, y=1.13, xanchor="left", yanchor="top", pad=dict(r=8, t=0),
            buttons=[
                dict(label="▶", method="animate", args=[None, _anim_cfg()]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ]
        )]

    def _anim_slider(n):
        every = max(1, n // 20)   # show at most ~20 tick labels
        return [dict(
            active=0, x=0.0, len=1.0, y=0, pad=dict(b=10, t=50),
            currentvalue=dict(prefix="Step: ", visible=True, xanchor="center",
                              font=dict(size=11, color="#64748B")),
            transition=dict(duration=int(frame_ms * 0.5), easing="linear"),
            steps=[dict(method="animate", args=[[str(i)], _imm_cfg()],
                        label=str(i) if i % every == 0 else "")
                   for i in range(n)],
        )]

    # ── animated figure (▶ Run mode) ──
    def make_animated_1d(steps):
        s0 = steps[0]
        frames = [go.Frame(data=[_agent_trace_1d(s), _best_trace_1d(s)],
                           traces=[1, 2], name=str(i))
                  for i, s in enumerate(steps)]
        fig = go.Figure(
            data=[go.Scatter(x=_xs1d.tolist(), y=_ys1d.tolist(), mode="lines",
                             name=fn_obj.name, line=dict(color=C_CURVE, width=2.5)),
                  _agent_trace_1d(s0), _best_trace_1d(s0)],
            frames=frames,
        )
        _L1 = {**_L, "margin": dict(l=12, r=12, t=44, b=60)}
        fig.update_layout(
            **_L1, height=430,
            title=dict(text=f"<b>{fn_obj.name}</b> — 1D landscape",
                       font=dict(size=14, color="#1E293B")),
            xaxis=dict(title="x", range=[b_min, b_max], fixedrange=True,
                       showgrid=True, gridcolor="#F1F5F9",
                       zeroline=True, zerolinecolor="#CBD5E1"),
            yaxis=dict(title="f(x)", range=[_y_lo, _y_hi], fixedrange=True,
                       showgrid=True, gridcolor="#F1F5F9"),
            updatemenus=_anim_menus(),
            sliders=_anim_slider(len(steps)),
        )
        return fig

    def make_animated_2d(steps):
        xs_g, ys_g, Z_g = _get_grid_2d()
        s0 = steps[0]
        frames = [go.Frame(data=[_agent_trace_2d(s), _best_trace_2d(s)],
                           traces=[1, 2], name=str(i))
                  for i, s in enumerate(steps)]
        fig = go.Figure(
            data=[go.Contour(x=xs_g, y=ys_g, z=Z_g, colorscale="RdYlBu_r",
                             contours=dict(coloring="heatmap", showlabels=False),
                             showscale=True, colorbar=dict(thickness=12, len=0.85)),
                  _agent_trace_2d(s0), _best_trace_2d(s0)],
            frames=frames,
        )
        _L2 = {**_L, "margin": dict(l=12, r=12, t=44, b=60)}
        fig.update_layout(
            **_L2, height=470,
            title=dict(text=f"<b>{fn_obj.name}</b> — 2D landscape",
                       font=dict(size=14, color="#1E293B")),
            xaxis=dict(title="x₁", range=[b_min, b_max], fixedrange=True, showgrid=False),
            yaxis=dict(title="x₂", range=[b_min, b_max], fixedrange=True, showgrid=False),
            updatemenus=_anim_menus(),
            sliders=_anim_slider(len(steps)),
        )
        return fig

    # ── static figure (⏭ Step mode) ──
    def make_static_1d(state):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_xs1d.tolist(), y=_ys1d.tolist(), mode="lines",
                                 name=fn_obj.name, line=dict(color=C_CURVE, width=2.5)))
        if state:
            fig.add_trace(_agent_trace_1d(state))
            fig.add_trace(_best_trace_1d(state))
        fig.update_layout(
            **_L, height=400,
            title=dict(text=f"<b>{fn_obj.name}</b> — 1D landscape",
                       font=dict(size=14, color="#1E293B")),
            xaxis=dict(title="x", range=[b_min, b_max], fixedrange=True,
                       showgrid=True, gridcolor="#F1F5F9",
                       zeroline=True, zerolinecolor="#CBD5E1"),
            yaxis=dict(title="f(x)", range=[_y_lo, _y_hi], fixedrange=True,
                       showgrid=True, gridcolor="#F1F5F9"),
        )
        return fig

    def make_static_2d(state):
        xs_g, ys_g, Z_g = _get_grid_2d()
        fig = go.Figure()
        fig.add_trace(go.Contour(x=xs_g, y=ys_g, z=Z_g, colorscale="RdYlBu_r",
                                 contours=dict(coloring="heatmap", showlabels=False),
                                 showscale=True, colorbar=dict(thickness=12, len=0.85)))
        if state:
            fig.add_trace(_agent_trace_2d(state))
            fig.add_trace(_best_trace_2d(state))
        fig.update_layout(
            **_L, height=440,
            title=dict(text=f"<b>{fn_obj.name}</b> — 2D landscape",
                       font=dict(size=14, color="#1E293B")),
            xaxis=dict(title="x₁", range=[b_min, b_max], fixedrange=True, showgrid=False),
            yaxis=dict(title="x₂", range=[b_min, b_max], fixedrange=True, showgrid=False),
        )
        return fig

    # — render —
    st.markdown('<div class="plot-box">', unsafe_allow_html=True)
    if all_steps is not None:
        fig = make_animated_1d(all_steps) if dims == 1 else make_animated_2d(all_steps)
    else:
        fig = make_static_1d(state) if dims == 1 else make_static_2d(state)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="opt_main_plot")
    st.markdown('</div>', unsafe_allow_html=True)

    # — convergence —
    hist = all_steps[-1]["history"] if all_steps else (state["history"] if state else None)
    if hist and len(hist) > 1:
        conv = go.Figure()
        conv.add_trace(go.Scatter(y=hist, mode="lines",
                                  line=dict(color=C_CONV, width=2),
                                  fill="tozeroy", fillcolor="rgba(5,150,105,0.08)"))
        conv.update_layout(**_L, height=165,
                           title=dict(text="<b>Convergence</b> — best value vs iteration",
                                      font=dict(size=13, color="#1E293B")),
                           xaxis=dict(title="Iteration", range=[0, n_iters], fixedrange=True,
                                      showgrid=True, gridcolor="#F1F5F9"),
                           yaxis=dict(title="f(best)", fixedrange=True,
                                      showgrid=True, gridcolor="#F1F5F9"),
                           showlegend=False)
        st.markdown('<div class="plot-box" style="margin-top:0.5rem">', unsafe_allow_html=True)
        st.plotly_chart(conv, use_container_width=True, config={"displayModeBar": False},
                        key="opt_conv_plot")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Convergence chart appears after running.")

    # — algorithm description —
    with st.expander(f"ℹ️  {algo_name}  ·  {fn_name}"):
        st.markdown(f"**{algo_name}:** {algo.description}")
        st.markdown(f"**{fn_name}:** {fn_obj.description}")

# ══════════════════════════ RIGHT: statistics ═════════════════════════════════

with stats_col:
    st.markdown('<div class="stats-panel">', unsafe_allow_html=True)
    st.markdown("**📊 Statistics**")

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

        st.metric("Fn evals", f"{state['n_evals']:,}")

        if "temperature" in state:
            st.metric("Temperature  T", f"{state['temperature']:.4g}")

        if len(state["values"]) > 1:
            st.divider()
            st.caption("**Population**")
            v = state["values"]
            ca, cb = st.columns(2)
            ca.metric("Mean", f"{np.mean(v):.4g}")
            cb.metric("Std",  f"{np.std(v):.4g}")
            ca.metric("Min",  f"{np.min(v):.4g}")
            cb.metric("Max",  f"{np.max(v):.4g}")

        # ── ACO pheromone matrix ──
        if algo_name == "Ant Colony Optimization" and "arch_weights" in state:
            st.divider()
            st.caption("**🐜 Pheromone Matrix**")
            w    = state["arch_weights"]
            arch = state["archive"]
            k, d = arch.shape
            if k <= 6:
                # Rows = archive rank, cols = dimensions
                # Cell color = pheromone weight; cell text = coordinate value
                z_mat   = np.tile(w.reshape(-1, 1), (1, d))
                txt_mat = [[f"{arch[i, j]:.3f}" for j in range(d)] for i in range(k)]
                col_hdr = [f"x{j+1}" for j in range(d)] if d > 1 else ["x"]
                row_hdr = [f"R{i+1}  φ={w[i]:.3f}" for i in range(k)]
                ph_fig  = go.Figure(go.Heatmap(
                    z=z_mat, x=col_hdr, y=row_hdr,
                    colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
                    showscale=False,
                    zmin=0, zmax=float(w.max()),
                    text=txt_mat, texttemplate="%{text}",
                    textfont=dict(size=10, color="#1E293B"),
                ))
                ph_fig.update_layout(
                    template="simple_white",
                    plot_bgcolor="#FFFFFF", paper_bgcolor="#F8FAFC",
                    height=max(130, 38 * k + 40),
                    margin=dict(l=4, r=4, t=28, b=4),
                    title=dict(text="φ = pheromone · value = position",
                               font=dict(size=9, color="#94A3B8"), x=0),
                    xaxis=dict(side="top", tickfont=dict(size=9)),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                    font=dict(size=9),
                )
                st.plotly_chart(ph_fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.caption("R1 = best (darkest = strongest pheromone).")
            else:
                st.info(
                    f"Colony size {k} > 6 — matrix too large to display. "
                    "Reduce colony size to ≤ 6 to see the matrix.",
                    icon="🐜"
                )

    st.markdown('</div>', unsafe_allow_html=True)

