"""
Лаб. 2. Метод ломаных (метод Пиявского).
Ищем глобальный минимум липшицевой функции f на отрезке [a, b].
"""

from __future__ import annotations

import math
import time
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# Разрешённые имена для безопасного разбора строки функции
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
    """
    Из строки вида 'x + sin(pi*x)' делает функцию f(x).
    Префикс 'f(x) =' можно писать, можно не писать.
    """
    s = expr.strip()
    s = re.sub(r"^f\s*\(\s*x\s*\)\s*=\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("^", "**")

    def f(x: float) -> float:
        return float(eval(s, {"__builtins__": {}}, {**SAFE_MATH, "x": float(x)}))

    # быстрая проверка, что выражение считается
    f(0.0)
    return f, s


def estimate_lipschitz(f, a: float, b: float, n: int = 2000) -> float:
    """
    Оценка константы Липшица L по сетке:
    берём максимум |f'| ≈ |Δf|/Δx на отрезке и чуть увеличиваем запас.
    """
    xs = np.linspace(a, b, n)
    ys = np.array([f(x) for x in xs])
    diffs = np.abs(np.diff(ys) / np.diff(xs))
    L = float(np.max(diffs))
    if L < 1e-12:
        L = 1.0
    return 1.1 * L  # небольшой запас, чтобы нижняя оценка не «пробивала» f


@dataclass
class Result:
    x_best: float
    f_best: float
    iterations: int
    elapsed_sec: float
    points_x: list[float]
    points_y: list[float]
    L: float


def piyavskii(f, a: float, b: float, eps: float, L: float | None = None, max_iter: int = 5000) -> Result:
    """
    Метод ломаных (Пиявский) для min f(x) на [a, b].

    Идея:
    - в каждой пробной точке x_i строим «уголок» вниз: y = f(x_i) - L*|x - x_i|
    - нижняя оценка R(x) = max по всем уголкам
    - следующая точка — где R(x) минимальна (обычно пересечение соседних уголков)
    - стоп, когда f_best - R_min <= eps
    """
    if a >= b:
        raise ValueError("Нужно a < b")

    t0 = time.perf_counter()
    if L is None:
        L = estimate_lipschitz(f, a, b)

    # стартовые точки — концы отрезка
    points = {a: f(a), b: f(b)}

    for it in range(1, max_iter + 1):
        xs = sorted(points.keys())

        # среди всех пересечений соседних уголков ищем минимум нижней оценки
        best_char = None  # (R_value, x_candidate)
        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = points[x1], points[x2]

            # точка пересечения двух соседних V-образных нижних оценок
            x_star = 0.5 * (x1 + x2) + (y1 - y2) / (2.0 * L)
            # на всякий случай держим внутри отрезка [x1, x2]
            x_star = min(max(x_star, x1), x2)

            R = 0.5 * (y1 + y2) - 0.5 * L * (x2 - x1)

            if best_char is None or R < best_char[0]:
                best_char = (R, x_star)

        assert best_char is not None
        R_min, x_new = best_char
        f_best = min(points.values())

        # критерий остановки по точности
        if f_best - R_min <= eps:
            x_best = min(points, key=lambda x: points[x])
            elapsed = time.perf_counter() - t0
            xs_out = sorted(points.keys())
            return Result(
                x_best=x_best,
                f_best=points[x_best],
                iterations=it,
                elapsed_sec=elapsed,
                points_x=xs_out,
                points_y=[points[x] for x in xs_out],
                L=L,
            )

        # добавляем новую пробную точку
        if x_new not in points:
            points[x_new] = f(x_new)

    x_best = min(points, key=lambda x: points[x])
    elapsed = time.perf_counter() - t0
    xs_out = sorted(points.keys())
    return Result(
        x_best=x_best,
        f_best=points[x_best],
        iterations=max_iter,
        elapsed_sec=elapsed,
        points_x=xs_out,
        points_y=[points[x] for x in xs_out],
        L=L,
    )


def build_broken_line(points_x, points_y, L, a, b, n_plot: int = 800):
    """Строит нижнюю оценку R(x) = max_i (y_i - L*|x - x_i|) на сетке для графика."""
    grid = np.linspace(a, b, n_plot)
    R = np.full_like(grid, -np.inf)
    for xi, yi in zip(points_x, points_y):
        R = np.maximum(R, yi - L * np.abs(grid - xi))
    return grid, R


def visualize(f, a, b, res: Result, title: str, save_path: str):
    """Рисует функцию, ломаную (нижнюю оценку) и найденный минимум."""
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
    """1D-срез функции Растригина — несколько локальных минимумов."""
    return x * x - 10.0 * math.cos(2.0 * math.pi * x) + 10.0


def main():
    # --- демо 1: функция из задания ---
    expr = "x + sin(pi*x)"
    f1, expr_clean = make_function(expr)
    a1, b1, eps = 0.0, 4.0, 0.01
    res1 = piyavskii(f1, a1, b1, eps)
    visualize(
        f1, a1, b1, res1,
        title=f"Метод ломаных: f(x)={expr_clean} на [{a1}, {b1}]",
        save_path="lab2_demo_sin.png",
    )
    print("=== Демо 1: f(x) = x + sin(pi*x) ===")
    print(f"отрезок: [{a1}, {b1}], eps={eps}, L≈{res1.L:.4f}")
    print(f"x* ≈ {res1.x_best:.6f}")
    print(f"f(x*) ≈ {res1.f_best:.6f}")
    print(f"итераций: {res1.iterations}")
    print(f"время: {res1.elapsed_sec:.4f} сек")
    print("график: lab2_demo_sin.png")
    print()

    # --- демо 2: Растригин (несколько локальных минимумов) ---
    a2, b2 = -5.0, 5.0
    res2 = piyavskii(rastrigin_1d, a2, b2, eps)
    visualize(
        rastrigin_1d, a2, b2, res2,
        title="Метод ломаных: 1D Растригин на [-5, 5]",
        save_path="lab2_demo_rastrigin.png",
    )
    print("=== Демо 2: Растригин (1D) ===")
    print(f"отрезок: [{a2}, {b2}], eps={eps}, L≈{res2.L:.4f}")
    print(f"x* ≈ {res2.x_best:.6f}")
    print(f"f(x*) ≈ {res2.f_best:.6f}")
    print(f"итераций: {res2.iterations}")
    print(f"время: {res2.elapsed_sec:.4f} сек")
    print("график: lab2_demo_rastrigin.png")


if __name__ == "__main__":
    main()
