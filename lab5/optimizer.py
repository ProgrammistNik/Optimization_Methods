import math
from collections import deque

import numpy as np


DEFAULTS = dict(
    mode="hybrid",
    initial_points=64,
    cycles=8,
    annealing_share=0.55,
    final_share=0.16,
    temperature_ratio=0.05,
    cooling_ratio=1e-7,
    tabu_radius=0.012,
    tabu_tenure=12,
    basin_tenure=4,
    local_step=0.025,
    minimum_step=5e-11,
)


class Budget:
    def __init__(self, f, bounds, max_nfe):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("bounds must be a finite array of shape (2, 2)")
        self.lo = bounds[:, 0]
        self.hi = bounds[:, 1]
        self.width = self.hi - self.lo
        if np.any(self.width <= 0) or not np.all(np.isfinite(self.width)):
            raise ValueError("bounds must have finite positive widths")
        if isinstance(max_nfe, bool) or not isinstance(max_nfe, (int, np.integer)) or max_nfe < 1:
            raise ValueError("max_nfe must be a positive integer")
        if not callable(f):
            raise TypeError("f must be callable")
        self.f = f
        self.max_nfe = int(max_nfe)
        self.nfe = 0
        self.best_f = math.inf
        self.best_x = None
        self.best_u = None
        self.history = []
        self.trajectory = []
        self.switches = []
        self.snapshots = []
        self.stats = dict(sa_uphill=0, tabu_uphill=0, tabu_rejected=0, aspirations=0, basins=0)

    @property
    def left(self):
        return self.max_nfe - self.nfe

    def eval(self, u):
        if not self.left:
            raise RuntimeError("evaluation budget exhausted")
        x = np.clip(self.lo + self.width * u, self.lo, self.hi)
        self.nfe += 1
        value = float(self.f(x.copy()))
        if not math.isfinite(value):
            raise ValueError("f must return finite values")
        if value < self.best_f:
            self.best_f = value
            self.best_x = x.copy()
            self.best_u = np.asarray(u, dtype=float).copy()
            self.trajectory.append((self.nfe, x.copy()))
        self.history.append(self.best_f)
        return value

    def stage(self, name):
        self.switches.append((self.nfe, name))

    def snap(self, name, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.snapshots.append((self.nfe, name, np.clip(self.lo + self.width * points, self.lo, self.hi)))

    def result(self):
        return dict(x=self.best_x.copy(), f=self.best_f, nfe=self.nfe,
                    history=self.history, switches=self.switches, snapshots=self.snapshots,
                    trajectory=self.trajectory, stats=self.stats)


def reflect(u):
    return 1.0 - np.abs(np.remainder(u, 2.0) - 1.0)


def forbidden(point, memory, radius):
    return any((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 < radius ** 2
               for other in memory)


def anneal(budget, rng, start, value, count, temperature, memory, p, refine=False):
    stop = min(budget.max_nfe, budget.nfe + count)
    length = stop - budget.nfe
    if not length:
        return start, value
    point = start.copy()
    best, best_value = point.copy(), value
    noise = rng.normal(size=(length, 2))
    jumps = rng.standard_cauchy(size=(length, 2))
    kinds = rng.random(length)
    uniforms = rng.random((length, 2))
    thresholds = rng.random(length)
    for i in range(length):
        fraction = i / max(1, length - 1)
        if refine:
            step = 0.02 * (5e-8 ** fraction)
            heat = temperature * (1e-18 ** fraction)
            proposal = reflect(point + step * noise[i])
        else:
            step = 0.18 * (0.015 ** fraction)
            heat = temperature * (p["cooling_ratio"] ** fraction)
            if kinds[i] < 0.05:
                proposal = uniforms[i]
            elif kinds[i] < 0.4:
                proposal = reflect(point + step * jumps[i])
            else:
                proposal = reflect(point + step * noise[i])
        record = budget.best_f
        candidate = budget.eval(proposal)
        blocked = forbidden(proposal, memory, p["tabu_radius"])
        if blocked and candidate >= record:
            budget.stats["tabu_rejected"] += 1
            continue
        if blocked:
            budget.stats["aspirations"] += 1
        delta = candidate - value
        if delta <= 0 or (heat > 0 and math.log(max(thresholds[i], 1e-300)) < -delta / heat):
            budget.stats["sa_uphill"] += int(delta > 0)
            point, value = proposal, candidate
            if value < best_value:
                best, best_value = point.copy(), value
    return best, best_value


def tabu_search(budget, rng, start, value, count, p):
    stop = min(budget.max_nfe, budget.nfe + count)
    point = start.copy()
    best, best_value = point.copy(), value
    step = p["local_step"]
    memory = deque(maxlen=p["tabu_tenure"])
    failures = 0
    trend = np.zeros(2)
    while budget.nfe < stop and step >= p["minimum_step"]:
        angle = rng.uniform(0, math.pi / 2)
        c, s = math.cos(angle), math.sin(angle)
        directions = np.array(((1, 0), (-1, 0), (0, 1), (0, -1),
                               (c, s), (-c, -s), (-s, c), (s, -c)))
        length = math.hypot(*trend)
        if length > 0:
            directions = np.vstack((directions, trend / length, -trend / length))
        neighbors = np.clip(point + step * directions, 0, 1)
        before = best_value
        previous = best.copy()
        chosen, chosen_value = None, math.inf
        for proposal in neighbors:
            if budget.nfe >= stop:
                break
            candidate = budget.eval(proposal)
            blocked = forbidden(proposal, memory, step * 0.3)
            if blocked and candidate >= best_value:
                budget.stats["tabu_rejected"] += 1
                continue
            if blocked:
                budget.stats["aspirations"] += 1
            if candidate < chosen_value:
                chosen, chosen_value = proposal.copy(), candidate
            if candidate < best_value:
                best, best_value = proposal.copy(), candidate
        memory.append(point.copy())
        if chosen is not None:
            budget.stats["tabu_uphill"] += int(chosen_value > value)
            point, value = chosen, chosen_value
        if best_value < before:
            trend = 0.6 * trend + best - previous
        failures = 0 if best_value < before else failures + 1
        if failures >= 2 or chosen is None:
            step *= 0.5
            point, value = best.copy(), best_value
            failures = 0
    return best, best_value


def validate_params(params):
    unknown = params.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    if p["mode"] not in ("hybrid", "sa", "tabu"):
        raise ValueError("mode must be hybrid, sa or tabu")
    for key in ("initial_points", "cycles", "tabu_tenure", "basin_tenure"):
        if isinstance(p[key], bool) or not isinstance(p[key], (int, np.integer)) or p[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("annealing_share", "final_share", "temperature_ratio", "cooling_ratio",
                "tabu_radius", "local_step", "minimum_step"):
        if not math.isfinite(p[key]) or not 0 < p[key] < 1:
            raise ValueError(f"{key} must lie strictly between zero and one")
    if p["minimum_step"] >= p["local_step"]:
        raise ValueError("minimum_step must be smaller than local_step")
    return p


def optimize(f, bounds, max_nfe=10_000, seed=None, constraints=None, **params):
    p = validate_params(params)
    if constraints is not None and len(constraints):
        raise ValueError("variant 7 supports rectangular bounds only")
    budget = Budget(f, bounds, max_nfe)
    rng = np.random.default_rng(seed)
    budget.stage("Начальная выборка")
    points = rng.random((min(p["initial_points"], max_nfe), 2))
    budget.snap("Начало", points)
    values = np.array([budget.eval(point) for point in points])
    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    temperature = max(spread, np.finfo(float).eps * max(1.0, abs(budget.best_f))) * p["temperature_ratio"]
    memory = deque(maxlen=p["basin_tenure"])
    reserve = int(budget.left * p["final_share"])
    end = max_nfe - reserve
    for cycle in range(p["cycles"]):
        count = (end - budget.nfe) // (p["cycles"] - cycle)
        if count <= 0:
            break
        stop = budget.nfe + count
        if cycle == 0:
            start, value = budget.best_u.copy(), budget.best_f
        else:
            candidates = rng.random((min(24, count), 2))
            ranked = sorted((budget.eval(u), tuple(u)) for u in candidates)
            available = [(v, u) for v, u in ranked if not forbidden(u, memory, p["tabu_radius"])]
            value, start = (available or ranked)[0]
            start = np.asarray(start)
        count = stop - budget.nfe
        if p["mode"] != "tabu":
            budget.stage(f"Отжиг {cycle + 1}")
            sa_count = count if p["mode"] == "sa" else int(count * p["annealing_share"])
            start, value = anneal(budget, rng, start, value, sa_count, temperature, memory, p)
            if cycle == 0:
                budget.snap("После отжига", [start])
        if p["mode"] != "sa" and budget.nfe < stop:
            budget.stage(f"Запреты {cycle + 1}")
            start, value = tabu_search(budget, rng, start, value, stop - budget.nfe, p)
            memory.append(start.copy())
            budget.stats["basins"] += 1
            if cycle == 0:
                budget.snap("После поиска с запретами", [start])
    if budget.left:
        budget.stage("Финальное уточнение")
        start, value = budget.best_u.copy(), budget.best_f
        if p["mode"] == "sa":
            anneal(budget, rng, start, value, budget.left, temperature * 1e-5, (), p, refine=True)
        else:
            tabu_search(budget, rng, start, value, budget.left, p)
    budget.snap("Конец", [budget.best_u])
    return budget.result()
