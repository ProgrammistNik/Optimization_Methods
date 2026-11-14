from __future__ import annotations

import argparse
import heapq
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BOUNDS = (-5.12, 5.12, -5.12, 5.12)


def rastrigin2d(x: float, y: float, a: float = 10.0) -> float:
    return x * x + y * y + 2.0 * a * (math.sin(math.pi * x) ** 2 + math.sin(math.pi * y) ** 2)


@dataclass
class OptResult:
    x: float
    y: float
    f: float
    elapsed_sec: float
    n_eval: int
    status: str


class EvalCounter:
    def __init__(self, fun, lower_bound=None):
        self.fun = fun
        self.lower_bound = lower_bound
        self.n = 0
        self.cache = {}

    def __call__(self, x: float, y: float) -> float:
        key = (x, y)
        if key not in self.cache:
            value = float(self.fun(x, y))
            self.n += 1
            if not math.isfinite(value):
                raise ValueError('Функция должна возвращать конечные значения во всей области.')
            if self.lower_bound is not None and value < self.lower_bound - 1e-12 * max(1.0, abs(self.lower_bound)):
                raise ValueError('Указанная нижняя граница опровергнута значением функции.')
            self.cache[key] = value
        return self.cache[key]


def validate_bounds(bounds):
    if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
        raise ValueError('Область должна содержать четыре конечных числа.')
    xmin, xmax, ymin, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError('Нижняя граница каждой координаты должна быть меньше верхней.')
    if not math.isfinite(xmax - xmin) or not math.isfinite(ymax - ymin):
        raise ValueError('Ширина области должна быть конечной.')


def nelder_mead_2d(fcount, x0, y0, bounds, step=0.35, max_iter=240):
    xmin, xmax, ymin, ymax = bounds

    def clip(x, y):
        return min(max(x, xmin), xmax), min(max(y, ymin), ymax)

    x0, y0 = clip(x0, y0)
    dx = min(step, 0.2 * (xmax - xmin))
    dy = min(step, 0.2 * (ymax - ymin))
    simplex = [
        (x0, y0),
        (x0 + dx if x0 + dx <= xmax else x0 - dx, y0),
        (x0, y0 + dy if y0 + dy <= ymax else y0 - dy),
    ]
    values = [fcount(*point) for point in simplex]
    for _ in range(max_iter):
        order = sorted(range(3), key=values.__getitem__)
        simplex = [simplex[index] for index in order]
        values = [values[index] for index in order]
        best, good, worst = simplex
        span = max(math.hypot(best[0] - point[0], best[1] - point[1]) for point in simplex[1:])
        if span <= 1e-9 and values[2] - values[0] <= 1e-13:
            break
        cx, cy = 0.5 * (best[0] + good[0]), 0.5 * (best[1] + good[1])
        reflected = clip(2.0 * cx - worst[0], 2.0 * cy - worst[1])
        fr = fcount(*reflected)
        if fr < values[0]:
            expanded = clip(cx + 2.0 * (reflected[0] - cx), cy + 2.0 * (reflected[1] - cy))
            fe = fcount(*expanded)
            simplex[2], values[2] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[1]:
            simplex[2], values[2] = reflected, fr
        else:
            outside = fr < values[2]
            anchor = reflected if outside else worst
            contracted = clip(cx + 0.5 * (anchor[0] - cx), cy + 0.5 * (anchor[1] - cy))
            fc = fcount(*contracted)
            if fc < (fr if outside else values[2]):
                simplex[2], values[2] = contracted, fc
            else:
                for index in (1, 2):
                    simplex[index] = clip(
                        best[0] + 0.5 * (simplex[index][0] - best[0]),
                        best[1] + 0.5 * (simplex[index][1] - best[1]),
                    )
                    values[index] = fcount(*simplex[index])
    index = min(range(3), key=values.__getitem__)
    return *simplex[index], values[index]


def piyavskii_candidates(g, a, b, eps=5e-4, max_iter=96, samples=32, count=3):
    if not a < b:
        return [(a, float(g(a)))]
    points = [(a + (b - a) * index / (samples - 1), 0.0) for index in range(samples)]
    points = [(x, float(g(x))) for x, _ in points]
    best = min(points, key=lambda point: point[1])
    slope = max(abs((right[1] - left[1]) / (right[0] - left[0])) for left, right in zip(points, points[1:]))
    lipschitz = max(2.0 * slope, 1e-12)

    def interval(left, right):
        lower = 0.5 * (left[1] + right[1] - lipschitz * (right[0] - left[0]))
        return lower, left, right

    intervals = [interval(left, right) for left, right in zip(points, points[1:])]
    heapq.heapify(intervals)
    for _ in range(max_iter):
        if not intervals:
            break
        lower, left, right = heapq.heappop(intervals)
        if best[1] - lower <= eps:
            break
        x = 0.5 * (left[0] + right[0]) + (left[1] - right[1]) / (2.0 * lipschitz)
        if not left[0] < x < right[0]:
            x = 0.5 * (left[0] + right[0])
        if not left[0] < x < right[0]:
            continue
        value = float(g(x))
        point = (x, value)
        points.append(point)
        if value < best[1]:
            best = point
        observed = max(abs((value - left[1]) / (x - left[0])), abs((right[1] - value) / (right[0] - x)))
        if 2.0 * observed > lipschitz:
            lipschitz = 2.0 * observed
            intervals = [interval(first, last) for _, first, last in intervals]
            heapq.heapify(intervals)
        heapq.heappush(intervals, interval(left, point))
        heapq.heappush(intervals, interval(point, right))
    candidates = []
    for point in sorted(points, key=lambda item: item[1]):
        if all(abs(point[0] - other[0]) > 0.06 * (b - a) for other in candidates):
            candidates.append(point)
        if len(candidates) == count:
            break
    return candidates


def piyavskii_1d(g, a, b, eps=5e-4, max_iter=96):
    return piyavskii_candidates(g, min(a, b), max(a, b), eps, max_iter, count=1)[0]


def section_candidates(fcount, point, direction, bounds, eps, budget, count):
    x, y, _ = point
    dx, dy = direction
    lower, upper = -math.inf, math.inf
    for coordinate, component, low, high in ((x, dx, *bounds[:2]), (y, dy, *bounds[2:])):
        if abs(component) > 1e-15:
            first, last = sorted(((low - coordinate) / component, (high - coordinate) / component))
            lower, upper = max(lower, first), min(upper, last)
    if upper - lower < 1e-14:
        return [point]

    def position(t):
        return min(max(x + t * dx, bounds[0]), bounds[1]), min(max(y + t * dy, bounds[2]), bounds[3])

    candidates = piyavskii_candidates(lambda t: fcount(*position(t)), lower, upper, eps, budget, count=count)
    return [(*position(t), value) for t, value in candidates]


def hybrid_optimize(
    fun,
    bounds=DEFAULT_BOUNDS,
    grid=10,
    top_k=8,
    angles=6,
    line_eps=5e-4,
    local_step=0.35,
    *,
    lower_bound=None,
    target_tol=1e-13,
    seed=0,
    random_points=32,
    line_budget=96,
    strategy='guarded',
):
    start = time.perf_counter()
    validate_bounds(bounds)
    for value, minimum in ((grid, 2), (top_k, 1), (angles, 1), (random_points, 0), (line_budget, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('Неверное целочисленное ограничение поиска.')
    for value in (line_eps, local_step, target_tol):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Точность и шаг должны быть положительными конечными числами.')
    if lower_bound is not None and not math.isfinite(lower_bound):
        raise ValueError('Известная нижняя граница должна быть конечной.')
    if strategy not in ('multistart', 'axes', 'guarded'):
        raise ValueError('Неизвестная стратегия поиска.')
    fcount = EvalCounter(fun, lower_bound)
    xmin, xmax, ymin, ymax = bounds
    points = [
        (xmin + (xmax - xmin) * i / (grid - 1), ymin + (ymax - ymin) * j / (grid - 1))
        for i in range(grid) for j in range(grid)
    ]
    rng = random.Random(seed)
    points.extend((rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(random_points))
    ranked = sorted([(*point, fcount(*point)) for point in points], key=lambda point: point[2])
    seeds = []
    for point in ranked:
        if all(math.hypot((point[0] - other[0]) / (xmax - xmin), (point[1] - other[1]) / (ymax - ymin)) > 0.06 for other in seeds):
            seeds.append(point)
        if len(seeds) == top_k:
            break
    best = ranked[0]

    def target_reached():
        return lower_bound is not None and abs(best[2] - lower_bound) <= target_tol

    def polish(point):
        return nelder_mead_2d(fcount, point[0], point[1], bounds, step=local_step)

    for initial in seeds:
        if target_reached():
            break
        candidate = polish(initial)
        best = min(best, candidate, key=lambda point: point[2])
        if target_reached():
            break
        if strategy == 'multistart':
            continue
        for direction in ((1.0, 0.0), (0.0, 1.0)):
            alternatives = section_candidates(fcount, candidate, direction, bounds, line_eps, line_budget, 1)
            candidate = min(candidate, *alternatives, key=lambda point: point[2])
        candidate = polish(candidate)
        best = min(best, candidate, key=lambda point: point[2])
        if target_reached():
            break
        if strategy == 'guarded':
            for index in range(angles):
                angle = math.pi * index / angles
                direction = math.cos(angle), math.sin(angle)
                alternatives = section_candidates(fcount, best, direction, bounds, line_eps, line_budget, 3)
                for alternative in alternatives:
                    candidate = polish(alternative)
                    best = min(best, candidate, key=lambda point: point[2])
                    if target_reached():
                        break
                if target_reached():
                    break
    status = 'target_reached' if target_reached() else 'search_complete'
    return OptResult(*best, time.perf_counter() - start, fcount.n, status)


def save_visualization(fun, result, bounds, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    xmin, xmax, ymin, ymax = bounds
    xs, ys = np.linspace(xmin, xmax, 260), np.linspace(ymin, ymax, 260)
    xx, yy = np.meshgrid(xs, ys)
    values = np.array([[fun(float(x), float(y)) for x in xs] for y in ys])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), dpi=170)
    contours = axes[0].contourf(xx, yy, values, levels=35, cmap='viridis')
    fig.colorbar(contours, ax=axes[0], label='f(x, y)', shrink=0.85)
    axes[0].plot(result.x, result.y, '*', color='#ff7043', markersize=14, markeredgecolor='white', label='Найденная точка')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Вся область поиска')
    radius = 0.08 * min(xmax - xmin, ymax - ymin)
    zoom_x = np.linspace(max(xmin, result.x - radius), min(xmax, result.x + radius), 180)
    zoom_y = np.linspace(max(ymin, result.y - radius), min(ymax, result.y + radius), 180)
    zoom_values = np.array([[fun(float(x), float(y)) for x in zoom_x] for y in zoom_y])
    axes[1].contourf(zoom_x, zoom_y, zoom_values, levels=35, cmap='viridis')
    axes[1].plot(result.x, result.y, '*', color='#ff7043', markersize=14, markeredgecolor='white')
    axes[1].set_title(f'Окрестность результата, f* = {result.f:.3e}')
    for axis in axes:
        axis.set(xlabel='x', ylabel='y', aspect='equal')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def battle_print(result):
    print('Битва методов оптимизации')
    print('Соколов Никита Константинович, К3339, поток 1.2')
    print('Гибрид: сечения Пиявского и локальный поиск Нелдера-Мида')
    print(f'x* = {result.x:.15g}')
    print(f'y* = {result.y:.15g}')
    print(f'f* = {result.f:.15g}')
    print(f'время поиска = {result.elapsed_sec:.6f} с')
    print(f'вызовов f = {result.n_eval}')
    print(f'статус = {result.status}')


def main():
    parser = argparse.ArgumentParser(description='Битва методов оптимизации, Растригин в двумерной области')
    parser.add_argument('--bounds', type=float, nargs=4, default=DEFAULT_BOUNDS, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    parser.add_argument('--grid', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--strategy', choices=('multistart', 'axes', 'guarded'), default='guarded')
    parser.add_argument('--no-target', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    result = hybrid_optimize(
        rastrigin2d, bounds=args.bounds, grid=args.grid, seed=args.seed,
        strategy=args.strategy, lower_bound=None if args.no_target else 0.0,
    )
    battle_print(result)
    if not args.no_plot:
        path = Path(__file__).with_name('demo_rastrigin2d.png')
        save_visualization(rastrigin2d, result, args.bounds, path, 'Растригин 2D. Сечения Пиявского и Нелдер-Мид')
        print(f'График сохранён: {path.name}')
        print('Время поиска не включает импорт графических библиотек и построение графика.')


if __name__ == '__main__':
    main()
