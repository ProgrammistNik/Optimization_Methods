"""
Эталонный набор тестовых функций для практической работы
«Битва методов оптимизации: имитация явлений природы».

Один и тот же файл лежит у студентов (student_template/functions.py)
и у преподавателя (judge/functions.py). На битве используется только
версия преподавателя.

Все функции минимизируются. Значения f* и x* проверены численно.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from numpy import abs, cos, exp, pi, sin, sqrt

Array = np.ndarray


@dataclass(frozen=True)
class TestFunction:
    name: str                      # машинное имя (используется в --func)
    title: str                     # название для отчёта
    f: Callable[[Array], float]    # сама функция f(x), x.shape == (2,)
    bounds: tuple                  # ((a1, b1), (a2, b2))
    f_star: float                  # значение глобального минимума
    x_star: tuple                  # все известные точки глобального минимума в D
    eps: float = 1e-4              # порог успеха для нормированной ошибки
    constraints: tuple = ()        # функции g(x) <= 0 (допустимость)
    shiftable: bool = False        # можно ли строить сдвинутые/повёрнутые варианты
    note: str = ""

    def __call__(self, x) -> float:
        return float(self.f(np.asarray(x, dtype=float)))

    @property
    def lower(self) -> Array:
        return np.array([b[0] for b in self.bounds], dtype=float)

    @property
    def upper(self) -> Array:
        return np.array([b[1] for b in self.bounds], dtype=float)

    def is_feasible(self, x, tol: float = 1e-8) -> bool:
        x = np.asarray(x, dtype=float)
        if np.any(x < self.lower - tol) or np.any(x > self.upper + tol):
            return False
        return all(g(x) <= tol for g in self.constraints)

    def error(self, fx: float) -> float:
        """Нормированная ошибка e = (f(x) - f*) / max(1, |f*|)."""
        return (fx - self.f_star) / max(1.0, abs(self.f_star))

    def distance(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(min(np.linalg.norm(x - np.asarray(p)) for p in self.x_star))


# ---------------------------------------------------------------- формулы
def rastrigin(x):
    return 20 + np.sum(x**2 - 10 * cos(2 * pi * x))


SCHWEFEL_C = 418.9828872724338


def schwefel(x):
    return SCHWEFEL_C * 2 - np.sum(x * sin(sqrt(abs(x))))


def holder_table(x):
    return -abs(sin(x[0]) * cos(x[1]) * exp(abs(1 - sqrt(x[0]**2 + x[1]**2) / pi)))


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def ackley(x):
    return (-20 * exp(-0.2 * sqrt(0.5 * (x[0]**2 + x[1]**2)))
            - exp(0.5 * (cos(2 * pi * x[0]) + cos(2 * pi * x[1]))) + np.e + 20)


def eggholder(x):
    return (-(x[1] + 47) * sin(sqrt(abs(x[1] + x[0] / 2 + 47)))
            - x[0] * sin(sqrt(abs(x[0] - (x[1] + 47)))))


def levy13(x):
    return (sin(3 * pi * x[0])**2
            + (x[0] - 1)**2 * (1 + sin(3 * pi * x[1])**2)
            + (x[1] - 1)**2 * (1 + sin(2 * pi * x[1])**2))


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def cross_in_tray(x):
    return -0.0001 * (abs(sin(x[0]) * sin(x[1])
                          * exp(abs(100 - sqrt(x[0]**2 + x[1]**2) / pi))) + 1)**0.1


def michalewicz(x, m=10):
    i = np.arange(1, x.size + 1)
    return -np.sum(sin(x) * sin(i * x**2 / pi)**(2 * m))


def goldstein_price(x):
    a = 1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1]
                                     + 6 * x[0] * x[1] + 3 * x[1]**2)
    b = 30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1]
                                         - 36 * x[0] * x[1] + 27 * x[1]**2)
    return a * b


def styblinski_tang(x):
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


def drop_wave(x):
    r2 = x[0]**2 + x[1]**2
    return -(1 + cos(12 * sqrt(r2))) / (0.5 * r2 + 2)


def schaffer2(x):
    return 0.5 + (sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2


def easom(x):
    return -cos(x[0]) * cos(x[1]) * exp(-(x[0] - pi)**2 - (x[1] - pi)**2)


def bukin6(x):
    return 100 * sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10)


def beale(x):
    return ((1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2
            + (2.625 - x[0] + x[0] * x[1]**3)**2)


def six_hump_camel(x):
    return ((4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2 + x[0] * x[1]
            + (-4 + 4 * x[1]**2) * x[1]**2)


def branin(x):
    return ((x[1] - 5.1 / (4 * pi**2) * x[0]**2 + 5 / pi * x[0] - 6)**2
            + 10 * (1 - 1 / (8 * pi)) * cos(x[0]) + 10)


def mishra_bird(x):
    return (sin(x[1]) * exp((1 - cos(x[0]))**2) + cos(x[0]) * exp((1 - sin(x[1]))**2)
            + (x[0] - x[1])**2)


def g_disk(x):            # x1^2 + x2^2 <= 2
    return x[0]**2 + x[1]**2 - 2


def g_mishra(x):          # (x1+5)^2 + (x2+5)^2 < 25
    return (x[0] + 5)**2 + (x[1] + 5)**2 - 25


# ---------------------------------------------------------------- реестр
_L = [
    TestFunction("rastrigin", "Растригина", rastrigin, ((-5.12, 5.12),) * 2, 0.0, ((0, 0),), shiftable=True),
    TestFunction("schwefel", "Швефеля", schwefel, ((-500, 500),) * 2, 0.0,
                 ((420.9687463, 420.9687463),), note="с точной константой 418.98288727 f* = 0"),
    TestFunction("holder_table", "Hölder table", holder_table, ((-10, 10),) * 2, -19.2085025678868,
                 ((8.05502347, 9.66459003), (-8.05502347, 9.66459003),
                  (8.05502347, -9.66459003), (-8.05502347, -9.66459003))),
    TestFunction("rosenbrock", "Розенброка", rosenbrock, ((-2, 2), (-1, 3)), 0.0, ((1, 1),), shiftable=True),
    TestFunction("ackley", "Экли", ackley, ((-5, 5),) * 2, 0.0, ((0, 0),), shiftable=True),
    TestFunction("eggholder", "Eggholder", eggholder, ((-512, 512),) * 2, -959.6406627208506,
                 ((512, 404.2318050),), note="минимум на границе области"),
    TestFunction("levy13", "Леви N.13", levy13, ((-10, 10),) * 2, 0.0, ((1, 1),), shiftable=True),
    TestFunction("himmelblau", "Химмельблау", himmelblau, ((-5, 5),) * 2, 0.0,
                 ((3, 2), (-2.80511809, 3.13131252), (-3.77931025, -3.28318599), (3.58442834, -1.84812653)),
                 shiftable=True),
    TestFunction("cross_in_tray", "Cross-in-tray", cross_in_tray, ((-10, 10),) * 2, -2.0626118708227397,
                 ((1.34940663, 1.34940663), (-1.34940663, 1.34940663),
                  (1.34940663, -1.34940663), (-1.34940663, -1.34940663))),
    TestFunction("michalewicz", "Михалевича (m = 10)", michalewicz, ((0, pi),) * 2, -1.8013034100985532,
                 ((2.20290552, 1.57079633),)),
    TestFunction("goldstein_price", "Гольдштейна–Прайса", goldstein_price, ((-2, 2),) * 2, 3.0, ((0, -1),)),
    TestFunction("styblinski_tang", "Стыбинского–Танга", styblinski_tang, ((-5, 5),) * 2, -78.33233140754285,
                 ((-2.90353403, -2.90353403),), shiftable=True),
    TestFunction("drop_wave", "Drop-wave", drop_wave, ((-5.12, 5.12),) * 2, -1.0, ((0, 0),), shiftable=True),
    TestFunction("schaffer2", "Шаффера N.2", schaffer2, ((-100, 100),) * 2, 0.0, ((0, 0),), shiftable=True),
    TestFunction("easom", "Изома", easom, ((-100, 100),) * 2, -1.0, ((pi, pi),), shiftable=True),
    TestFunction("bukin6", "Букина N.6", bukin6, ((-15, -5), (-3, 3)), 0.0, ((-10, 1),), eps=1e-2,
                 note="очень узкий хребет, порог успеха ослаблен"),
    TestFunction("beale", "Била", beale, ((-4.5, 4.5),) * 2, 0.0, ((3, 0.5),), shiftable=True),
    TestFunction("six_hump_camel", "Шестигорбый верблюд", six_hump_camel, ((-3, 3), (-2, 2)),
                 -1.0316284534898774, ((0.08984201, -0.7126564), (-0.08984201, 0.7126564)), shiftable=True),
    TestFunction("branin", "Бранина", branin, ((-5, 10), (0, 15)), 0.39788735772973816,
                 ((-pi, 12.275), (pi, 2.275), (3 * pi, 2.475)), shiftable=True),
    TestFunction("rosenbrock_disk", "Розенброка с ограничением x1²+x2²≤2", rosenbrock, ((-1.5, 1.5),) * 2,
                 0.0, ((1, 1),), constraints=(g_disk,), note="вариант 16"),
    TestFunction("mishra_bird", "Mishra's Bird с ограничением", mishra_bird, ((-10, 0), (-6.5, 0)),
                 -106.76453674926476, ((-3.13024681, -1.58214217),), constraints=(g_mishra,),
                 note="резервный вариант"),
]
FUNCTIONS: dict[str, TestFunction] = {t.name: t for t in _L}

# Соответствие вариантов из методических указаний функциям
VARIANTS = {1: "rastrigin", 2: "schwefel", 3: "holder_table", 4: "rosenbrock", 5: "ackley",
            6: "eggholder", 7: "levy13", 8: "himmelblau", 9: "cross_in_tray", 10: "michalewicz",
            11: "goldstein_price", 12: "styblinski_tang", 13: "drop_wave", 14: "schaffer2",
            15: "easom", 16: "rosenbrock_disk", 17: "bukin6", 18: "beale", 19: "six_hump_camel",
            20: "branin", 21: "mishra_bird"}


# ---------------------------------------------------- скрытые варианты для финала
def make_variant(base: str | TestFunction, seed: int, shrink: float = 0.6) -> TestFunction:
    """
    Сдвинутая и повёрнутая версия функции: g(x) = f(x0 + R(x - c)),
    где x0 — исходная точка минимума, R — поворот на случайный угол,
    c — новая точка минимума (случайно во внутренних shrink*100 % области).
    Допускаются только функции с shiftable=True: их f* — глобальный минимум на всей
    плоскости, поэтому у варианта то же значение f*.
    """
    t = FUNCTIONS[base] if isinstance(base, str) else base
    if not t.shiftable:
        raise ValueError(f"{t.name}: сдвиг не поддерживается")
    rng = np.random.default_rng(seed)
    lo, hi = t.lower, t.upper
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * shrink
    c = rng.uniform(mid - half, mid + half)
    phi = rng.uniform(0, 2 * pi)
    R = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
    x0 = np.asarray(t.x_star[0], dtype=float)
    f0 = t.f

    def g(x, f0=f0, x0=x0, R=R, c=c):
        return f0(x0 + R @ (np.asarray(x, dtype=float) - c))

    pts = [tuple(float(v) for v in c + R.T @ (np.asarray(p) - x0)) for p in t.x_star]
    pts = [p for p in pts if np.all(np.asarray(p) >= lo) and np.all(np.asarray(p) <= hi)] or [tuple(float(v) for v in c)]
    return replace(t, name=f"{t.name}~{seed}", title=f"{t.title} (сдвиг/поворот, seed={seed})",
                   f=g, x_star=tuple(pts), shiftable=False, note="скрытый вариант")


def get_function(spec: str) -> TestFunction:
    """'rastrigin' — исходная функция; 'rastrigin~17' — скрытый вариант с seed=17."""
    if "~" in spec:
        base, s = spec.split("~", 1)
        return make_variant(base, int(s))
    return FUNCTIONS[spec]


if __name__ == "__main__":
    # Самопроверка: f(x*) == f* для всех функций и вариантов
    bad = 0
    for t in list(FUNCTIONS.values()) + [make_variant(n, 1) for n, t in FUNCTIONS.items() if t.shiftable]:
        for p in t.x_star:
            e = abs(t(p) - t.f_star)
            ok = e < 1e-6 and t.is_feasible(p, tol=1e-6)
            bad += not ok
            print(f"{'OK ' if ok else 'ERR'} {t.name:22s} f(x*)={t(p):.10g}  f*={t.f_star:.10g}")
    print("Ошибок:", bad)
