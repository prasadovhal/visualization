Build a polished interactive web app that can be deployed on **GitHub Pages** to visualize optimization algorithms. This is a teaching/demo project only; do not use anything from my research.

### Goal

Let users visually see how different optimization algorithms search for the optimum of a mathematical function.

### Algorithms

Implement:

1. Random Walk
2. Monte Carlo Random Search
3. Metropolis Monte Carlo
4. Simulated Annealing
5. Genetic Algorithm
6. Ant Colony Optimization
7. Black Hole Algorithm

### Optimization functions

Support both **1D and 2D functions**.

Include standard benchmark functions such as:

* Sphere
* Rastrigin
* Ackley
* Rosenbrock
* Easom
* Himmelblau
* Beale

Only show functions compatible with the selected dimensionality.

### User controls

Allow the user to select:

* Algorithm
* Optimization function
* 1D or 2D mode
* Search-space min/max
* Population/agent count where applicable
* Number of iterations
* Algorithm-specific parameters
* Random seed

Provide:

* **Run automatically**
* **Pause**
* **Next Step**
* **Reset**
* Speed control

### Visualization

For **1D**:

* Plot the objective function curve.
* Show current candidate(s) as moving points.
* Highlight the best-so-far solution.
* Show the search history.

For **2D**:

* Show contour/heatmap of the objective function.
* Plot agents/candidates on top.
* Animate how they move through the search space.
* Highlight current global best.
* Optionally show trails/history.

For population-based algorithms, visually distinguish the population and best candidate.

### Algorithm information

Show a small panel containing:

* Current iteration
* Best x / (x,y)
* Best objective value
* Number of function evaluations
* Current temperature for SA/Metropolis if relevant
* Population statistics where useful

Also show a small plain-English explanation of what the selected algorithm is doing.

### Architecture

Create this as a **static frontend suitable for GitHub Pages**.

Preferred stack:

* React
* TypeScript
* Vite
* Plotly.js or another interactive plotting library

Do not require a backend.

Keep optimization logic separate from UI code.

Use an interface similar to:

```ts
interface Optimizer {
  initialize(config: OptimizerConfig): OptimizerState;
  step(state: OptimizerState): OptimizerState;
  getBest(state: OptimizerState): Candidate;
}
```

Each algorithm should implement the same interface so the UI can switch algorithms easily.

### Important

The **step() operation must perform exactly one logical optimization iteration**, because the same implementation must power both:

* automatic animation
* manual step-by-step execution

Make runs reproducible using the selected random seed.

### UI layout

Use a clean educational layout:

* Left sidebar → configuration
* Center → large interactive optimization plot
* Right panel → algorithm state/statistics
* Bottom → convergence graph showing best objective value vs iteration

Make it responsive and visually polished.

### GitHub Pages

Configure the project so it can be deployed directly using GitHub Pages.

Include:

* GitHub Actions deployment workflow
* correct Vite base-path handling
* README with local setup and deployment instructions

### Code quality

* Modular TypeScript
* Clear folder structure
* Reusable optimizer interface
* Comments explaining algorithm logic
* Avoid unnecessary dependencies
* Add basic unit tests for objective functions and optimizer steps

Create the complete project, run it locally, fix TypeScript/build errors, run tests, and ensure `npm run build` succeeds before considering the task complete.
