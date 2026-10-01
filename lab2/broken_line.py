from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


SAFE_MATH = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}


def make_function(expr: str):
    s = expr.strip()
    s = re.sub(r"^f\s*\(\s*x\s*\)\s*=\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("^", "**")

    def f(x: float) -> float:
        return float(eval(s, {"__builtins__": {}}, {**SAFE_MATH, "x": float(x)}))

    f(0.0)
    return f, s


def estimate_lipschitz(
    f,
    a: float,
    b: float,
    n0: int = 500,
    max_n: int = 100000,
    tol: float = 0.01,
    safety: float = 1.1,
) -> float:
    n = n0
    L_prev = None
    L_best = 0.0
    while n <= max_n:
        xs = np.linspace(a, b, n)
        ys = np.array([f(x) for x in xs])
        L_n = float(np.max(np.abs(np.diff(ys) / np.diff(xs))))
        L_best = max(L_best, L_n)
        if L_prev is not None and abs(L_n - L_prev) <= tol * max(L_best, 1e-12):
            break
        L_prev = L_n
        n *= 2
    if L_best < 1e-12:
        L_best = 1.0
    return safety * L_best


@dataclass
class Result:
    x_best: float
    f_best: float
    iterations: int
    elapsed_sec: float
    points_x: list[float]
    points_y: list[float]
    L: float
    status: str


def piyavskii(f, a: float, b: float, eps: float, L: float | None = None, max_iter: int = 5000) -> Result:
    if a >= b:
        raise ValueError("Нужно a < b")

    t0 = time.perf_counter()
    if L is None:
        L = estimate_lipschitz(f, a, b)

    points = {a: f(a), b: f(b)}
    n_eval = 2

    for _ in range(1, max_iter + 1):
        xs = sorted(points.keys())
        best_char = None
        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = points[x1], points[x2]
            x_star = 0.5 * (x1 + x2) + (y1 - y2) / (2.0 * L)
            x_star = min(max(x_star, x1), x2)
            R = 0.5 * (y1 + y2) - 0.5 * L * (x2 - x1)
            if best_char is None or R < best_char[0]:
                best_char = (R, x_star)

        assert best_char is not None
        R_min, x_new = best_char

        if x_new not in points:
            points[x_new] = f(x_new)
            n_eval += 1

        f_best = min(points.values())
        if f_best - R_min <= eps:
            x_best = min(points, key=lambda x: points[x])
            xs_out = sorted(points.keys())
            return Result(
                x_best=x_best,
                f_best=points[x_best],
                iterations=n_eval,
                elapsed_sec=time.perf_counter() - t0,
                points_x=xs_out,
                points_y=[points[x] for x in xs_out],
                L=L,
                status="converged",
            )

    x_best = min(points, key=lambda x: points[x])
    xs_out = sorted(points.keys())
    return Result(
        x_best=x_best,
        f_best=points[x_best],
        iterations=n_eval,
        elapsed_sec=time.perf_counter() - t0,
        points_x=xs_out,
        points_y=[points[x] for x in xs_out],
        L=L,
        status="max_iter",
    )


def build_broken_line(points_x, points_y, L, a, b, n_plot: int = 800):
    grid = np.linspace(a, b, n_plot)
    R = np.full_like(grid, -np.inf)
    for xi, yi in zip(points_x, points_y):
        R = np.maximum(R, yi - L * np.abs(grid - xi))
    return grid, R


def visualize(f, a, b, res: Result, title: str, save_path: str):
    xs = np.linspace(a, b, 1000)
    ys = np.array([f(x) for x in xs])
    grid, R = build_broken_line(res.points_x, res.points_y, res.L, a, b)

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label="f(x)", color="steelblue", linewidth=2)
    plt.plot(grid, R, label="нижняя оценка (ломаная)", color="orange", linewidth=1.5)
    plt.scatter(res.points_x, res.points_y, s=18, color="gray", label="пробные точки", zorder=3)
    plt.scatter([res.x_best], [res.f_best], s=80, color="red", zorder=4, label="найденный минимум")
    plt.axvline(res.x_best, color="red", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def rastrigin_1d(x: float) -> float:
    return x * x - 10.0 * math.cos(2.0 * math.pi * x) + 10.0


def main():
    eps = 0.01

    a1, b1 = -5.0, 5.0
    res1 = piyavskii(rastrigin_1d, a1, b1, eps)
    visualize(
        rastrigin_1d, a1, b1, res1,
        title="Метод ломаных: 1D Растригин на [-5, 5]",
        save_path="lab2_demo_rastrigin.png",
    )
    print("=== Демо 1: Растригин (1D), несколько локальных минимумов ===")
    print(f"отрезок: [{a1}, {b1}], eps={eps}, L≈{res1.L:.4f}")
    print(f"x* ≈ {res1.x_best:.6f}")
    print(f"f(x*) ≈ {res1.f_best:.6f}")
    print(f"вызовов f: {res1.iterations}")
    print(f"status: {res1.status}")
    print(f"время: {res1.elapsed_sec:.4f} сек")
    print("график: lab2_demo_rastrigin.png")
    print()

    expr = "x + sin(pi*x)"
    f2, expr_clean = make_function(expr)
    a2, b2 = 0.0, 4.0
    res2 = piyavskii(f2, a2, b2, eps)
    visualize(
        f2, a2, b2, res2,
        title=f"Метод ломаных: f(x)={expr_clean} на [{a2}, {b2}]",
        save_path="lab2_demo_sin.png",
    )
    print("=== Демо 2: f(x) = x + sin(pi*x) ===")
    print(f"отрезок: [{a2}, {b2}], eps={eps}, L≈{res2.L:.4f}")
    print(f"x* ≈ {res2.x_best:.6f}")
    print(f"f(x*) ≈ {res2.f_best:.6f}")
    print(f"вызовов f: {res2.iterations}")
    print(f"status: {res2.status}")
    print(f"время: {res2.elapsed_sec:.4f} сек")
    print("график: lab2_demo_sin.png")


if __name__ == "__main__":
    main()
