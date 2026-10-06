from __future__ import annotations

import time
from dataclasses import dataclass

import sympy as sp


EPS = 1e-9
VARIANT = 7


@dataclass
class Candidate:
    number: int
    point: dict[sp.Symbol, sp.Expr]
    multipliers: dict[sp.Symbol, sp.Expr]
    f_value: sp.Expr
    residuals: list[sp.Expr]
    feasible: bool


def build_problem():
    x, y, z = sp.symbols("x y z", real=True)
    lam = sp.symbols("lambda", real=True)
    variables = [x, y, z]
    multipliers = [lam]
    f = x + y + z
    constraints = [x**2 + y**2 + z**2 - 9]
    return f, constraints, variables, multipliers


def lagrangian(f, constraints, multipliers):
    return f + sum(lam * h for lam, h in zip(multipliers, constraints))


def stationarity_equations(L, variables, constraints):
    return [sp.diff(L, v) for v in variables] + constraints


def is_feasible(residuals: list[sp.Expr]) -> bool:
    for r in residuals:
        val = complex(sp.N(r))
        if abs(val.imag) > EPS:
            return False
        if abs(val.real) > EPS:
            return False
    return True


def numeric_value(expr: sp.Expr) -> float:
    val = complex(sp.N(expr))
    if abs(val.imag) > EPS:
        raise ValueError(f"ожидалось действительное значение, получено {expr}")
    return val.real


def dedupe_solutions(raw: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for sol in raw:
        key_parts = []
        for k in sorted(sol.keys(), key=str):
            key_parts.append((str(k), sp.srepr(sp.simplify(sol[k]))))
        key = tuple(key_parts)
        if key not in seen:
            seen.add(key)
            unique.append(sol)
    return unique


def solve_lagrange(
    f: sp.Expr,
    constraints: list[sp.Expr],
    variables: list[sp.Symbol],
    multipliers: list[sp.Symbol],
) -> tuple[sp.Expr, list[sp.Expr], list[Candidate]]:
    L = lagrangian(f, constraints, multipliers)
    equations = stationarity_equations(L, variables, constraints)
    unknowns = variables + multipliers

    raw = sp.solve(equations, unknowns, dict=True)
    if not raw:
        raw = []
        for sol in sp.nonlinsolve(equations, unknowns):
            raw.append(dict(zip(unknowns, sol)))

    solutions = dedupe_solutions(raw)
    candidates: list[Candidate] = []

    for i, sol in enumerate(solutions, start=1):
        point = {v: sp.simplify(sol[v]) for v in variables}
        mult = {m: sp.simplify(sol[m]) for m in multipliers}
        residuals = [sp.simplify(h.subs(point)) for h in constraints]
        feasible = is_feasible(residuals)
        f_val = sp.simplify(f.subs(point))
        candidates.append(
            Candidate(
                number=i,
                point=point,
                multipliers=mult,
                f_value=f_val,
                residuals=residuals,
                feasible=feasible,
            )
        )

    return L, equations, candidates


def pick_extrema(candidates: list[Candidate]) -> tuple[Candidate | None, Candidate | None]:
    ok = [c for c in candidates if c.feasible]
    if not ok:
        return None, None
    ok.sort(key=lambda c: numeric_value(c.f_value))
    return ok[0], ok[-1]


def print_problem(f, constraints, L, equations, variables, multipliers):
    print("=" * 68)
    print(f"Лабораторная работа №3 · метод множителей Лагранжа · вариант {VARIANT}")
    print("=" * 68)
    print("\nЦелевая функция:")
    print(f"  f(x,y,z) = {f}")
    print("\nОграничения (вид h = 0):")
    for h in constraints:
        print(f"  {h} = 0")
    print("\nФункция Лагранжа:")
    print(f"  L = {sp.expand(L)}")
    print("\nУравнения стационарности:")
    for i, eq in enumerate(equations, start=1):
        print(f"  ({i}) {sp.expand(eq)} = 0")


def print_candidates(candidates: list[Candidate], variables: list[sp.Symbol], multipliers: list[sp.Symbol]):
    print("\nКандидаты:")
    if not candidates:
        print("  (действительных решений системы не найдено)")
        return
    for c in candidates:
        coords = ", ".join(f"{v} = {c.point[v]}" for v in variables)
        lams = ", ".join(f"{m} = {c.multipliers[m]}" for m in multipliers)
        res = ", ".join(str(r) for r in c.residuals)
        print(f"\nКандидат {c.number}:")
        print(f"  точка: {coords}")
        print(f"  множители: {lams}")
        print(f"  значение f: {c.f_value} ≈ {numeric_value(c.f_value):.6f}")
        print(f"  невязки ограничений: {res}")
        print(f"  допустимость: {'да' if c.feasible else 'нет'}")


def format_point(point: dict[sp.Symbol, sp.Expr], variables: list[sp.Symbol]) -> str:
    parts = [f"{numeric_value(point[v]):.6f}" for v in variables]
    return "(" + ", ".join(parts) + ")"


def print_conclusion(
    c_min: Candidate | None,
    c_max: Candidate | None,
    variables: list[sp.Symbol],
):
    print("\nИтог:")
    if c_min is None:
        print("  допустимых кандидатов нет")
        return
    print(
        f"  глобальный минимум: f = {c_min.f_value} ≈ {numeric_value(c_min.f_value):.6f} "
        f"в точке {format_point(c_min.point, variables)}"
    )
    assert c_max is not None
    print(
        f"  глобальный максимум: f = {c_max.f_value} ≈ {numeric_value(c_max.f_value):.6f} "
        f"в точке {format_point(c_max.point, variables)}"
    )
    print(
        "  основание: допустимое множество — сфера x²+y²+z²=9 (компакт), "
        "f непрерывна; сравнение значений f во всех кандидатах Лагранжа."
    )
    print(
        "  аналитически: x=y=z=-√3 → f=-3√3; x=y=z=√3 → f=3√3."
    )


def main() -> None:
    t0 = time.perf_counter()
    f, constraints, variables, multipliers = build_problem()
    L, equations, candidates = solve_lagrange(f, constraints, variables, multipliers)
    c_min, c_max = pick_extrema(candidates)

    print_problem(f, constraints, L, equations, variables, multipliers)
    print_candidates(candidates, variables, multipliers)
    print_conclusion(c_min, c_max, variables)
    print(f"\nвремя: {time.perf_counter() - t0:.4f} сек")


if __name__ == "__main__":
    main()
