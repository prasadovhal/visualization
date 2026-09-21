"""
All optimizers share the same interface:
    initialize(config, fn) -> state dict
    step(state, config, fn) -> new state dict

state keys (always present):
    iteration       int
    candidates      np.ndarray  shape (n_agents, dims)
    values          np.ndarray  shape (n_agents,)
    best_pos        np.ndarray  shape (dims,)
    best_val        float
    history         list[float]  best value per iteration
    n_evals         int
    rng             np.random.Generator
    temperature     float  (SA / Metropolis only)
"""

import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def _clip(x, bounds):
    return np.clip(x, bounds[0], bounds[1])


def _rand_pop(n, dims, bounds, rng):
    return rng.uniform(bounds[0], bounds[1], (n, dims))


def _eval_all(pop, fn):
    return np.array([fn(p) for p in pop])


def _best(values):
    return int(np.argmin(values))


# ─── Random Walk ──────────────────────────────────────────────────────────────

class RandomWalk:
    name = "Random Walk"
    description = (
        "A single agent wanders the search space by adding Gaussian noise "
        "to its current position at every step. It always moves (even to worse "
        "positions), but remembers the best location ever visited."
    )
    defaults = {"step_size": 0.05}

    def initialize(self, config, fn):
        rng = np.random.default_rng(config["seed"])
        dims, bounds = config["dims"], config["bounds"]
        pos = rng.uniform(bounds[0], bounds[1], dims)
        val = fn(pos)
        return dict(
            iteration=0,
            candidates=pos[np.newaxis].copy(),
            values=np.array([val]),
            best_pos=pos.copy(), best_val=val,
            history=[val], n_evals=1, rng=rng,
            current_pos=pos.copy(), current_val=val,
        )

    def step(self, state, config, fn):
        rng, bounds = state["rng"], config["bounds"]
        sigma = config.get("step_size", 0.05) * (bounds[1] - bounds[0])
        new_pos = _clip(state["current_pos"] + rng.normal(0, sigma, config["dims"]), bounds)
        new_val = fn(new_pos)
        best_pos = new_pos.copy() if new_val < state["best_val"] else state["best_pos"].copy()
        best_val = new_val if new_val < state["best_val"] else state["best_val"]
        return {**state,
                "iteration": state["iteration"] + 1,
                "candidates": new_pos[np.newaxis].copy(),
                "values": np.array([new_val]),
                "best_pos": best_pos, "best_val": best_val,
                "history": state["history"] + [best_val],
                "n_evals": state["n_evals"] + 1,
                "current_pos": new_pos.copy(), "current_val": new_val}


# ─── Monte Carlo Random Search ────────────────────────────────────────────────

class MonteCarlo:
    name = "Monte Carlo"
    description = (
        "Each iteration one random candidate is uniformly sampled from the entire "
        "search space and evaluated. If it beats the current best it becomes the "
        "new best. Simple, memoryless, and surprisingly useful as a baseline."
    )
    defaults = {}

    def initialize(self, config, fn):
        rng = np.random.default_rng(config["seed"])
        dims, bounds = config["dims"], config["bounds"]
        pos = rng.uniform(bounds[0], bounds[1], dims)
        val = fn(pos)
        return dict(
            iteration=0, candidates=pos[np.newaxis].copy(), values=np.array([val]),
            best_pos=pos.copy(), best_val=val,
            history=[val], n_evals=1, rng=rng,
        )

    def step(self, state, config, fn):
        rng, bounds = state["rng"], config["bounds"]
        pos = rng.uniform(bounds[0], bounds[1], config["dims"])
        val = fn(pos)
        best_val = val if val < state["best_val"] else state["best_val"]
        best_pos = pos.copy() if val < state["best_val"] else state["best_pos"].copy()
        return {**state,
                "iteration": state["iteration"] + 1,
                "candidates": pos[np.newaxis].copy(), "values": np.array([val]),
                "best_pos": best_pos, "best_val": best_val,
                "history": state["history"] + [best_val],
                "n_evals": state["n_evals"] + 1}


# ─── Metropolis Monte Carlo ───────────────────────────────────────────────────

class Metropolis:
    name = "Metropolis Monte Carlo"
    description = (
        "A Markov chain sampler. A proposal is drawn near the current point. "
        "Improvements are always accepted; worse proposals are accepted with "
        "probability exp(−ΔE / T), allowing escape from local minima. "
        "Temperature T is fixed throughout."
    )
    defaults = {"temperature": 1.0, "step_size": 0.05}

    def initialize(self, config, fn):
        rng = np.random.default_rng(config["seed"])
        dims, bounds = config["dims"], config["bounds"]
        pos = rng.uniform(bounds[0], bounds[1], dims)
        val = fn(pos)
        T = config.get("temperature", 1.0)
        return dict(
            iteration=0, candidates=pos[np.newaxis].copy(), values=np.array([val]),
            best_pos=pos.copy(), best_val=val, history=[val], n_evals=1, rng=rng,
            temperature=T, current_pos=pos.copy(), current_val=val,
        )

    def step(self, state, config, fn):
        rng, bounds = state["rng"], config["bounds"]
        T = config.get("temperature", 1.0)
        sigma = config.get("step_size", 0.05) * (bounds[1] - bounds[0])
        new_pos = _clip(state["current_pos"] + rng.normal(0, sigma, config["dims"]), bounds)
        new_val = fn(new_pos)
        delta = new_val - state["current_val"]
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            cur_pos, cur_val = new_pos.copy(), new_val
        else:
            cur_pos, cur_val = state["current_pos"].copy(), state["current_val"]
        best_val = cur_val if cur_val < state["best_val"] else state["best_val"]
        best_pos = cur_pos.copy() if cur_val < state["best_val"] else state["best_pos"].copy()
        return {**state,
                "iteration": state["iteration"] + 1,
                "candidates": cur_pos[np.newaxis].copy(), "values": np.array([cur_val]),
                "best_pos": best_pos, "best_val": best_val,
                "history": state["history"] + [best_val],
                "n_evals": state["n_evals"] + 1,
                "temperature": T, "current_pos": cur_pos, "current_val": cur_val}


# ─── Simulated Annealing ──────────────────────────────────────────────────────

class SimulatedAnnealing:
    name = "Simulated Annealing"
    description = (
        "Metropolis Monte Carlo with a cooling temperature: T = T₀ × αⁿ. "
        "High T early on → wide exploration and frequent uphill moves. "
        "As T → 0 the algorithm becomes greedy and converges on the best found."
    )
    defaults = {"initial_temp": 10.0, "cooling_rate": 0.95, "step_size": 0.1}

    def initialize(self, config, fn):
        rng = np.random.default_rng(config["seed"])
        dims, bounds = config["dims"], config["bounds"]
        T0 = config.get("initial_temp", 10.0)
        pos = rng.uniform(bounds[0], bounds[1], dims)
        val = fn(pos)
        return dict(
            iteration=0, candidates=pos[np.newaxis].copy(), values=np.array([val]),
            best_pos=pos.copy(), best_val=val, history=[val], n_evals=1, rng=rng,
            temperature=T0, current_pos=pos.copy(), current_val=val,
        )

    def step(self, state, config, fn):
        rng, bounds = state["rng"], config["bounds"]
        T0 = config.get("initial_temp", 10.0)
        alpha = config.get("cooling_rate", 0.95)
        sigma = config.get("step_size", 0.1) * (bounds[1] - bounds[0])
        iteration = state["iteration"] + 1
        T = T0 * (alpha ** iteration)
        new_pos = _clip(state["current_pos"] + rng.normal(0, sigma, config["dims"]), bounds)
        new_val = fn(new_pos)
        delta = new_val - state["current_val"]
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            cur_pos, cur_val = new_pos.copy(), new_val
        else:
            cur_pos, cur_val = state["current_pos"].copy(), state["current_val"]
        best_val = cur_val if cur_val < state["best_val"] else state["best_val"]
        best_pos = cur_pos.copy() if cur_val < state["best_val"] else state["best_pos"].copy()
        return {**state,
                "iteration": iteration,
                "candidates": cur_pos[np.newaxis].copy(), "values": np.array([cur_val]),
                "best_pos": best_pos, "best_val": best_val,
                "history": state["history"] + [best_val],
                "n_evals": state["n_evals"] + 1,
                "temperature": T, "current_pos": cur_pos, "current_val": cur_val}


# ─── Genetic Algorithm ────────────────────────────────────────────────────────

class GeneticAlgorithm:
    name = "Genetic Algorithm"
    description = (
        "Maintains a population of candidate solutions. Each iteration: two parents "
        "are chosen via 2-tournament selection; blend crossover creates a child; "
        "Gaussian mutation randomly perturbs genes; the child replaces the worst "
        "individual. Over generations the population evolves toward better regions."
    )
    defaults = {"population_size": 20, "mutation_rate": 0.15, "crossover_rate": 0.8}

    def initialize(self, config, fn):
        rng = np.random.default_rng(config["seed"])
        n = config.get("population_size", 20)
        pop = _rand_pop(n, config["dims"], config["bounds"], rng)
        vals = _eval_all(pop, fn)
        bi = _best(vals)
        return dict(
            iteration=0, candidates=pop, values=vals,
            best_pos=pop[bi].copy(), best_val=float(vals[bi]),
            history=[float(vals[bi])], n_evals=n, rng=rng,
        )

    def step(self, state, config, fn):
        rng, bounds = state["rng"], config["bounds"]
        n = len(state["candidates"])
        dims = config["dims"]
        mut_rate = config.get("mutation_rate", 0.15)
        cx_rate = config.get("crossover_rate", 0.8)
        pop = state["candidates"].copy()
        vals = state["values"].copy()

        # 2-tournament selection
        def tournament():
            i1, i2 = rng.choice(n, 2, replace=False)
            return pop[i1].copy() if vals[i1] < vals[i2] else pop[i2].copy()

        p1, p2 = tournament(), tournament()

        # Blend crossover BLX-0.5
        if rng.random() < cx_rate:
            alpha = 0.5
            lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
            hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
            child = _clip(rng.uniform(lo, hi), bounds)
        else:
            child = p1.copy()

        # Gaussian mutation per-gene
        sigma = 0.1 * (bounds[1] - bounds[0])
        mask = rng.random(dims) < mut_rate
        child[mask] = _clip(child[mask] + rng.normal(0, sigma, mask.sum()), bounds)

        child_val = fn(child)

        # Replace worst
        worst = int(np.argmax(vals))
        pop[worst] = child
        vals[worst] = child_val

        bi = _best(vals)
        best_val = float(vals[bi]) if float(vals[bi]) < state["best_val"] else state["best_val"]
        best_pos = pop[bi].copy() if float(vals[bi]) < state["best_val"] else state["best_pos"].copy()
        return {**state,
                "iteration": state["iteration"] + 1,
                "candidates": pop, "values": vals,
                "best_pos": best_pos, "best_val": best_val,
                "history": state["history"] + [best_val],
                "n_evals": state["n_evals"] + 1}


# ─── Ant Colony Optimization (continuous) ────────────────────────────────────

class AntColony:
    name = "Ant Colony Optimization"
    description = (
        "Continuous ACO (ACOR). Maintains a ranked archive of the best solutions "
        "found so far. Each iteration every ant samples a new solution from a "
        "Gaussian mixture centred on archive members (higher-ranked members get "
        "more weight). New solutions that beat archive entries replace them."
    )
    defaults = {"n_ants": 10, "archive_size": 10, "xi": 0.85, "q": 0.5}

    def initialize(self, config, fn):
        rng = np.random.default_rng(config["seed"])
        k = config.get("archive_size", 10)
        q = config.get("q", 0.5)
        archive = _rand_pop(k, config["dims"], config["bounds"], rng)
        arch_vals = _eval_all(archive, fn)
        order = np.argsort(arch_vals)
        archive, arch_vals = archive[order], arch_vals[order]
        ranks = np.arange(1, k + 1, dtype=float)
        w = np.exp(-((ranks - 1) ** 2) / (2 * q**2 * k**2))
        w /= w.sum()
        return dict(
            iteration=0, candidates=archive.copy(), values=arch_vals.copy(),
            best_pos=archive[0].copy(), best_val=float(arch_vals[0]),
            history=[float(arch_vals[0])], n_evals=k, rng=rng,
            archive=archive.copy(), arch_vals=arch_vals.copy(),
            arch_weights=w.copy(),
        )

    def step(self, state, config, fn):
        rng, bounds = state["rng"], config["bounds"]
        dims = config["dims"]
        k = config.get("archive_size", 10)
        n = config.get("n_ants", 10)
        q = config.get("q", 0.5)
        xi = config.get("xi", 0.85)
        archive = state["archive"].copy()
        arch_vals = state["arch_vals"].copy()

        # Gaussian weights favouring better-ranked archive members
        ranks = np.arange(1, k + 1, dtype=float)
        w = np.exp(-((ranks - 1) ** 2) / (2 * q**2 * k**2))
        w /= w.sum()

        new_sols, new_vals = [], []
        for _ in range(n):
            l = rng.choice(k, p=w)
            sigma = np.array([
                xi * np.sum(np.abs(archive[l, d] - archive[:, d])) / max(k - 1, 1)
                for d in range(dims)
            ])
            sigma = np.maximum(sigma, 1e-6)
            sol = _clip(rng.normal(archive[l], sigma), bounds)
            new_sols.append(sol)
            new_vals.append(fn(sol))

        all_sols = np.vstack([archive, np.array(new_sols)])
        all_vals = np.concatenate([arch_vals, np.array(new_vals)])
        order = np.argsort(all_vals)[:k]
        archive, arch_vals = all_sols[order], all_vals[order]

        best_val = float(arch_vals[0]) if float(arch_vals[0]) < state["best_val"] else state["best_val"]
        best_pos = archive[0].copy() if float(arch_vals[0]) < state["best_val"] else state["best_pos"].copy()
        return {**state,
                "iteration": state["iteration"] + 1,
                "candidates": np.array(new_sols), "values": np.array(new_vals),
                "best_pos": best_pos, "best_val": best_val,
                "history": state["history"] + [best_val],
                "n_evals": state["n_evals"] + n,
                "archive": archive, "arch_vals": arch_vals,
                "arch_weights": w.copy()}


# ─── Black Hole Algorithm ─────────────────────────────────────────────────────

class BlackHole:
    name = "Black Hole Algorithm"
    description = (
        "Stars (candidates) are attracted toward the best solution, the black hole. "
        "Each step every star moves: x_new = x + rand × (x_BH − x). "
        "Stars that enter the event horizon (come too close to the black hole) "
        "are absorbed and replaced with a new random star, maintaining diversity."
    )
    defaults = {"population_size": 20}

    def initialize(self, config, fn):
        rng = np.random.default_rng(config["seed"])
        n = config.get("population_size", 20)
        pop = _rand_pop(n, config["dims"], config["bounds"], rng)
        vals = _eval_all(pop, fn)
        bi = _best(vals)
        return dict(
            iteration=0, candidates=pop, values=vals,
            best_pos=pop[bi].copy(), best_val=float(vals[bi]),
            history=[float(vals[bi])], n_evals=n, rng=rng,
        )

    def step(self, state, config, fn):
        rng, bounds = state["rng"], config["bounds"]
        dims = config["dims"]
        pop = state["candidates"].copy()
        vals = state["values"].copy()
        n = len(pop)
        n_evals = 0

        bh_idx = _best(vals)
        bh_pos = pop[bh_idx].copy()
        bh_val = float(vals[bh_idx])

        # Move every star toward the black hole
        for i in range(n):
            if i == bh_idx:
                continue
            r = rng.random()
            pop[i] = _clip(pop[i] + r * (bh_pos - pop[i]), bounds)
            vals[i] = fn(pop[i])
            n_evals += 1

        # Event horizon: fixed 10% of search diameter
        event_horizon = 0.10 * (bounds[1] - bounds[0]) * np.sqrt(dims)
        for i in range(n):
            if i == bh_idx:
                continue
            if np.linalg.norm(pop[i] - bh_pos) < event_horizon:
                pop[i] = _clip(rng.uniform(bounds[0], bounds[1], dims), bounds)
                vals[i] = fn(pop[i])
                n_evals += 1

        bi = _best(vals)
        best_val = float(vals[bi]) if float(vals[bi]) < state["best_val"] else state["best_val"]
        best_pos = pop[bi].copy() if float(vals[bi]) < state["best_val"] else state["best_pos"].copy()
        return {**state,
                "iteration": state["iteration"] + 1,
                "candidates": pop, "values": vals,
                "best_pos": best_pos, "best_val": best_val,
                "history": state["history"] + [best_val],
                "n_evals": state["n_evals"] + n_evals}


# ─── registry ─────────────────────────────────────────────────────────────────

ALGORITHMS = {
    "Random Walk":             RandomWalk(),
    "Monte Carlo":             MonteCarlo(),
    "Metropolis Monte Carlo":  Metropolis(),
    "Simulated Annealing":     SimulatedAnnealing(),
    "Genetic Algorithm":       GeneticAlgorithm(),
    "Ant Colony Optimization": AntColony(),
    "Black Hole":              BlackHole(),
}
