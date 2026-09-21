import numpy as np


class ObjFn:
    def __init__(self, name, fn, dims, bounds, description):
        self.name = name
        self.fn = fn          # callable: np.ndarray -> float
        self.dims = dims      # list of supported dimensionalities, e.g. [1, 2]
        self.bounds = bounds  # (default_min, default_max)
        self.description = description


def sphere(x):
    return float(np.sum(x**2))


def rastrigin(x):
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    n = len(x)
    a = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    b = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return float(a + b + 20 + np.e)


def rosenbrock(x):
    if len(x) == 1:
        return float((1.0 - x[0]) ** 2)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def easom(x):
    xi = x[0]
    return float(-np.cos(xi) * np.exp(-((xi - np.pi) ** 2)))


def himmelblau(x):
    xi, yi = x[0], x[1]
    return float((xi**2 + yi - 11) ** 2 + (xi + yi**2 - 7) ** 2)


def beale(x):
    xi, yi = x[0], x[1]
    t1 = (1.5 - xi + xi * yi) ** 2
    t2 = (2.25 - xi + xi * yi**2) ** 2
    t3 = (2.625 - xi + xi * yi**3) ** 2
    return float(t1 + t2 + t3)


FUNCTIONS = {
    "Sphere":     ObjFn("Sphere",     sphere,     [1, 2], (-5.0, 5.0),   "Smooth bowl. Global min = 0 at origin."),
    "Rastrigin":  ObjFn("Rastrigin",  rastrigin,  [1, 2], (-5.12, 5.12), "Highly multimodal. Global min = 0 at origin."),
    "Ackley":     ObjFn("Ackley",     ackley,     [1, 2], (-5.0, 5.0),   "Flat outer region, sharp global min = 0 at origin."),
    "Rosenbrock": ObjFn("Rosenbrock", rosenbrock, [1, 2], (-2.0, 2.0),   "Banana-shaped valley. Global min = 0 at (1, …, 1)."),
    "Easom":      ObjFn("Easom",      easom,      [1],    (-10.0, 10.0), "1D only. Sharp global min ≈ −1 at x = π ≈ 3.14159."),
    "Himmelblau": ObjFn("Himmelblau", himmelblau, [2],    (-5.0, 5.0),   "2D only. Four global minima = 0 (e.g. at (3, 2))."),
    "Beale":      ObjFn("Beale",      beale,      [2],    (-4.5, 4.5),   "2D only. Global min = 0 at (3, 0.5)."),
}
