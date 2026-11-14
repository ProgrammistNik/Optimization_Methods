from __future__ import annotations

import math
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def rastrigin2d(x: float, y: float, a: float = 10.0) -> float:
    return (
        2.0 * a
        + (x * x - a * math.cos(2.0 * math.pi * x))
        + (y * y - a * math.cos(2.0 * math.pi * y))
    )


@dataclass
class OptResult:
    x: float
    y: float
    f: float
    elapsed_sec: float
    n_eval: int
    status: str


class EvalCounter:
    def __init__(self, fun):
        self.fun = fun
        self.n = 0

    def __call__(self, x: float, y: float) -> float:
        self.n += 1
        return float(self.fun(x, y))


def estimate_lipschitz_1d(g, a: float, b: float, n0: int = 128) -> float:
    n = n0
    best = 0.0
    prev = None
    while n <= 2048:
        xs = np.linspace(a, b, n)
        ys = np.array([g(float(t)) for t in xs], dtype=float)
        dens = np.abs(np.diff(ys) / np.diff(xs))
        cur = float(np.max(dens)) if dens.size else 0.0
        best = max(best, cur)
        if prev is not None and abs(cur - prev) <= 0.01 * max(best, 1e-12):
            break
        prev = cur
        n *= 2
    if best < 1e-12:
        best = 1.0
    return 1.15 * best


def piyavskii_1d(g, a: float, b: float, eps: float = 1e-4, max_iter: int = 800) -> tuple[float, float]:
    if a > b:
        a, b = b, a
    if abs(b - a) < 1e-15:
        return a, float(g(a))
    L = estimate_lipschitz_1d(g, a, b)
    pts = [(a, float(g(a))), (b, float(g(b)))]
    best_x, best_f = min(pts, key=lambda p: p[1])
    for _ in range(max_iter):
        pts.sort(key=lambda p: p[0])
        best_char = None
        best_val = float("inf")
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            if x2 - x1 < 1e-15:
                continue
            intersection = 0.5 * (x1 + x2) + (y1 - y2) / (2.0 * L)
            if intersection <= x1 or intersection >= x2:
                intersection = 0.5 * (x1 + x2)
            char = 0.5 * (y1 + y2) - 0.5 * L * (x2 - x1)
            if char < best_val:
                best_val = char
                best_char = intersection
        if best_char is None:
            break
        y_new = float(g(best_char))
        pts.append((best_char, y_new))
        if y_new < best_f:
            best_f = y_new
            best_x = best_char
        if best_f - best_val <= eps:
            break
    return best_x, best_f


def nelder_mead_2d(
    fcount: EvalCounter,
    x0: float,
    y0: float,
    bounds: tuple[float, float, float, float],
    step: float = 0.35,
    max_iter: int = 350,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> tuple[float, float, float]:
    xmin, xmax, ymin, ymax = bounds

    def clip(x: float, y: float) -> tuple[float, float]:
        return min(max(x, xmin), xmax), min(max(y, ymin), ymax)

    def val(p: tuple[float, float]) -> float:
        return fcount(p[0], p[1])

    p0 = clip(x0, y0)
    p1 = clip(x0 + step, y0)
    p2 = clip(x0, y0 + step)
    simplex = [p0, p1, p2]
    scores = [val(p) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(3), key=lambda i: scores[i])
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        best, good, worst = simplex
        f_best, f_good, f_worst = scores
        span = math.hypot(best[0] - worst[0], best[1] - worst[1])
        if span < 1e-10 and abs(f_worst - f_best) < 1e-12:
            break
        cx = 0.5 * (best[0] + good[0])
        cy = 0.5 * (best[1] + good[1])
        xr = clip(cx + alpha * (cx - worst[0]), cy + alpha * (cy - worst[1]))
        fr = val(xr)
        if fr < f_best:
            xe = clip(cx + gamma * (xr[0] - cx), cy + gamma * (xr[1] - cy))
            fe = val(xe)
            if fe < fr:
                simplex[2], scores[2] = xe, fe
            else:
                simplex[2], scores[2] = xr, fr
        elif fr < f_good:
            simplex[2], scores[2] = xr, fr
        else:
            if fr < f_worst:
                xc = clip(cx + rho * (xr[0] - cx), cy + rho * (xr[1] - cy))
            else:
                xc = clip(cx + rho * (worst[0] - cx), cy + rho * (worst[1] - cy))
            fc = val(xc)
            if fc < f_worst:
                simplex[2], scores[2] = xc, fc
            else:
                simplex[1] = clip(
                    best[0] + sigma * (good[0] - best[0]),
                    best[1] + sigma * (good[1] - best[1]),
                )
                simplex[2] = clip(
                    best[0] + sigma * (worst[0] - best[0]),
                    best[1] + sigma * (worst[1] - best[1]),
                )
                scores[1] = val(simplex[1])
                scores[2] = val(simplex[2])

    order = sorted(range(3), key=lambda i: scores[i])
    bx, by = simplex[order[0]]
    return bx, by, scores[order[0]]


def hybrid_optimize(
    fun,
    bounds: tuple[float, float, float, float] = (-5.12, 5.12, -5.12, 5.12),
    grid: int = 10,
    top_k: int = 8,
    angles: int = 6,
    line_eps: float = 5e-4,
    local_step: float = 0.45,
) -> OptResult:
    xmin, xmax, ymin, ymax = bounds
    fcount = EvalCounter(fun)
    t0 = time.perf_counter()

    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(ymin, ymax, grid)
    seeds: list[tuple[float, float, float]] = []
    for x in xs:
        for y in ys:
            fx = fcount(float(x), float(y))
            seeds.append((float(x), float(y), fx))
    seeds.sort(key=lambda t: t[2])
    seeds = seeds[:top_k]

    refined: list[tuple[float, float, float]] = []
    for sx, sy, sf in seeds:
        best_x, best_y, best_f = sx, sy, sf
        if best_f <= 1e-12:
            refined.append((best_x, best_y, best_f))
            break
        for k in range(angles):
            ang = math.pi * k / angles
            c, s = math.cos(ang), math.sin(ang)

            def ray_pos(t: float) -> tuple[float, float]:
                return sx + t * c, sy + t * s

            cand: list[float] = []
            if abs(c) > 1e-12:
                cand.append((xmin - sx) / c)
                cand.append((xmax - sx) / c)
            if abs(s) > 1e-12:
                cand.append((ymin - sy) / s)
                cand.append((ymax - sy) / s)
            ts: list[float] = []
            for val in cand:
                px, py = ray_pos(val)
                if xmin - 1e-9 <= px <= xmax + 1e-9 and ymin - 1e-9 <= py <= ymax + 1e-9:
                    ts.append(val)
            if len(ts) < 2:
                continue
            t_lo, t_hi = min(ts), max(ts)
            if t_hi - t_lo < 1e-12:
                continue

            def g(t: float) -> float:
                px, py = ray_pos(t)
                px = min(max(px, xmin), xmax)
                py = min(max(py, ymin), ymax)
                return fcount(px, py)

            t_star, _ = piyavskii_1d(g, t_lo, t_hi, eps=line_eps, max_iter=250)
            px, py = ray_pos(t_star)
            px = min(max(px, xmin), xmax)
            py = min(max(py, ymin), ymax)
            fv = fcount(px, py)
            if fv < best_f:
                best_x, best_y, best_f = px, py, fv
            if best_f <= 1e-12:
                break
        refined.append((best_x, best_y, best_f))
        if best_f <= 1e-12:
            break

    refined.sort(key=lambda t: t[2])
    final_candidates = refined[: max(4, top_k // 2)]

    best_x, best_y, best_f = final_candidates[0]
    if best_f > 1e-12:
        for sx, sy, _ in final_candidates:
            lx, ly, lf = nelder_mead_2d(
                fcount, sx, sy, bounds, step=local_step, max_iter=220
            )
            if lf < best_f:
                best_x, best_y, best_f = lx, ly, lf
            if best_f <= 1e-12:
                break

    elapsed = time.perf_counter() - t0
    status = "ok"
    if abs(best_x) < 1e-3 and abs(best_y) < 1e-3 and abs(best_f) < 1e-2:
        status = "near_global"
    return OptResult(best_x, best_y, best_f, elapsed, fcount.n, status)


def save_visualization(
    fun,
    result: OptResult,
    bounds: tuple[float, float, float, float],
    path: str,
    title: str,
) -> None:
    xmin, xmax, ymin, ymax = bounds
    n = 220
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(fun)(X, Y)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    cs = ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax, shrink=0.86, label="f(x, y)")
    ax.plot([result.x], [result.y], marker="*", markersize=16, color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def battle_print(result: OptResult) -> None:
    print("=" * 64)
    print("Битва методов оптимизации")
    print("Гибрид: сечения методом ломаных (Пиявский) и Нелдер-Мид")
    print("=" * 64)
    print(f"x* = {result.x:.12g}")
    print(f"y* = {result.y:.12g}")
    print(f"f(x*, y*) = {result.f:.12g}")
    print(f"время = {result.elapsed_sec:.6f} с")
    print(f"вызовов f = {result.n_eval}")
    print(f"status = {result.status}")


def main() -> None:
    bounds = (-5.12, 5.12, -5.12, 5.12)
    result = hybrid_optimize(rastrigin2d, bounds=bounds)
    battle_print(result)
    out = __file__.replace("hybrid_optimize.py", "demo_rastrigin2d.png")
    save_visualization(
        rastrigin2d,
        result,
        bounds,
        out,
        "Растригин 2D: найденная точка гибридного метода",
    )
    print(f"график: {out}")


if __name__ == "__main__":
    main()
