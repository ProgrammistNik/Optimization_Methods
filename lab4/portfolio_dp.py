from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable


EPS = 1e-9
Q = 10  # шаг квантования состояния, д.е.


@dataclass(frozen=True)
class State:
    v1: int
    v2: int
    v3: int
    cash: int

    def wealth(self) -> int:
        return self.v1 + self.v2 + self.v3 + self.cash


@dataclass(frozen=True)
class Action:
    k1: int
    k2: int
    k3: int

    def as_tuple(self) -> tuple[int, int, int]:
        return self.k1, self.k2, self.k3


@dataclass(frozen=True)
class StageMarket:
    probs: tuple[float, float, float]
    r1: tuple[float, float, float]
    r2: tuple[float, float, float]
    r3: tuple[float, float, float]


@dataclass
class ProblemData:
    initial: State
    packet: tuple[int, int, int]
    mins: tuple[int, int, int]
    commission: tuple[float, float, float]
    stages: list[StageMarket]
    w0: int


def default_problem() -> ProblemData:
    initial = State(100, 800, 400, 600)
    stages = [
        StageMarket((0.6, 0.3, 0.1), (1.20, 1.05, 0.80), (1.10, 1.02, 0.95), (1.07, 1.03, 1.00)),
        StageMarket((0.3, 0.2, 0.5), (1.40, 1.05, 0.60), (1.15, 1.00, 0.90), (1.01, 1.00, 1.00)),
        StageMarket((0.4, 0.4, 0.2), (1.15, 1.05, 0.70), (1.12, 1.01, 0.94), (1.05, 1.01, 1.00)),
    ]
    return ProblemData(
        initial=initial,
        packet=(25, 200, 100),
        mins=(30, 150, 100),
        commission=(0.04, 0.07, 0.05),
        stages=stages,
        w0=initial.wealth(),
    )


def quantize(state: State) -> State:
    def q(x: float) -> int:
        return int(round(x / Q) * Q)

    return State(q(state.v1), q(state.v2), q(state.v3), q(state.cash))


def apply_action(state: State, action: Action, data: ProblemData) -> State | None:
    vals = [float(state.v1), float(state.v2), float(state.v3)]
    ks = list(action.as_tuple())
    cash = float(state.cash)

    for i in range(3):
        if ks[i] < 0:
            vol = (-ks[i]) * data.packet[i]
            cash += vol * (1.0 - data.commission[i])
            vals[i] -= vol
    for i in range(3):
        if ks[i] > 0:
            vol = ks[i] * data.packet[i]
            cash -= vol * (1.0 + data.commission[i])
            vals[i] += vol

    if cash < -EPS:
        return None
    for i in range(3):
        if vals[i] < data.mins[i] - EPS:
            return None
    return quantize(State(int(round(vals[0])), int(round(vals[1])), int(round(vals[2])), int(round(cash))))


def apply_market(state: State, stage: StageMarket, scenario: int) -> State:
    return quantize(
        State(
            int(round(state.v1 * stage.r1[scenario])),
            int(round(state.v2 * stage.r2[scenario])),
            int(round(state.v3 * stage.r3[scenario])),
            state.cash,
        )
    )


def feasible_actions(state: State, data: ProblemData) -> list[Action]:
    sell_max = []
    for i, v in enumerate((state.v1, state.v2, state.v3)):
        ms = int((v - data.mins[i]) // data.packet[i])
        sell_max.append(min(max(ms, 0), 2))

    buy_max = []
    for i in range(3):
        unit = data.packet[i] * (1.0 + data.commission[i])
        bm = int(state.cash // unit) if unit > 0 else 0
        buy_max.append(min(max(bm, 0), 2))

    actions: list[Action] = []
    for k1, k2, k3 in product(
        range(-sell_max[0], buy_max[0] + 1),
        range(-sell_max[1], buy_max[1] + 1),
        range(-sell_max[2], buy_max[2] + 1),
    ):
        # не более двух активов за этап
        if sum(1 for k in (k1, k2, k3) if k != 0) > 2:
            continue
        action = Action(k1, k2, k3)
        if apply_action(state, action, data) is not None:
            actions.append(action)
    return actions


def criterion_expectation(values: list[float], probs: tuple[float, float, float]) -> float:
    return sum(p * v for p, v in zip(probs, values))


def criterion_wald(values: list[float], probs: tuple[float, float, float]) -> float:
    return min(values)


def criterion_optimistic(values: list[float], probs: tuple[float, float, float]) -> float:
    return max(values)


def criterion_hurwicz(
    values: list[float], probs: tuple[float, float, float], alpha: float = 0.6
) -> float:
    return alpha * max(values) + (1.0 - alpha) * min(values)


CRITERIA: dict[str, Callable] = {
    "expectation": criterion_expectation,
    "wald": criterion_wald,
    "optimistic": criterion_optimistic,
    "hurwicz": criterion_hurwicz,
}


@dataclass
class SolveResult:
    criterion: str
    best_income: float
    expected_final_wealth: float
    policy: list[Action]
    path_states: list[State]


class BellmanSolver:
    def __init__(self, data: ProblemData, criterion_name: str = "expectation"):
        self.data = data
        self.criterion_name = criterion_name
        self.criterion = CRITERIA[criterion_name]
        self.policy_cache: dict[tuple[int, State], Action] = {}
        self.value_cache: dict[tuple[int, State], float] = {}

    def value(self, stage_idx: int, state: State) -> float:
        key = (stage_idx, state)
        if key in self.value_cache:
            return self.value_cache[key]

        if stage_idx >= len(self.data.stages):
            val = float(state.wealth() - self.data.w0)
            self.value_cache[key] = val
            return val

        stage = self.data.stages[stage_idx]
        best_val = float("-inf")
        best_action: Action | None = None

        for action in feasible_actions(state, self.data):
            after = apply_action(state, action, self.data)
            assert after is not None
            scenario_vals = [
                self.value(stage_idx + 1, apply_market(after, stage, sc)) for sc in range(3)
            ]
            score = self.criterion(scenario_vals, stage.probs)
            if score > best_val + EPS:
                best_val = score
                best_action = action

        if best_action is not None:
            self.policy_cache[key] = best_action
        self.value_cache[key] = best_val
        return best_val

    def solve(self) -> SolveResult:
        start = quantize(self.data.initial)
        best = self.value(0, start)
        policy: list[Action] = []
        path = [start]
        state = start

        for stage_idx in range(len(self.data.stages)):
            action = self.policy_cache[(stage_idx, state)]
            policy.append(action)
            after = apply_action(state, action, self.data)
            assert after is not None
            stage = self.data.stages[stage_idx]
            sc = max(range(3), key=lambda i: stage.probs[i])
            state = apply_market(after, stage, sc)
            path.append(state)

        return SolveResult(
            criterion=self.criterion_name,
            best_income=best,
            expected_final_wealth=best + self.data.w0,
            policy=policy,
            path_states=path,
        )


def simulate_policy(
    data: ProblemData, policy: list[Action], scenarios: list[int]
) -> tuple[list[State], float]:
    state = quantize(data.initial)
    path = [state]
    for stage_idx, action in enumerate(policy):
        after = apply_action(state, action, data)
        if after is None:
            raise RuntimeError("недопустимое управление")
        state = apply_market(after, data.stages[stage_idx], scenarios[stage_idx])
        path.append(state)
    return path, float(state.wealth() - data.w0)


def format_action(action: Action, data: ProblemData) -> str:
    names = ("ЦБ1", "ЦБ2", "Деп")
    parts = []
    for name, k, d in zip(names, action.as_tuple(), data.packet):
        if k == 0:
            continue
        op = "купить" if k > 0 else "продать"
        parts.append(f"{op} {name} на {abs(k) * d}")
    return "; ".join(parts) if parts else "без сделок"


def demo_stage3_fragment(data: ProblemData) -> None:
    print("\n=== Фрагмент ручной проверки: этап 3 ===")
    sample = quantize(State(100, 800, 400, 600))
    stage = data.stages[2]
    best = float("-inf")
    best_a = None
    checked = 0
    for action in feasible_actions(sample, data):
        checked += 1
        after = apply_action(sample, action, data)
        assert after is not None
        scen = [apply_market(after, stage, sc).wealth() - data.w0 for sc in range(3)]
        score = criterion_expectation(scen, stage.probs)
        if score > best:
            best = score
            best_a = action
    print(f"состояние: {sample}")
    print(f"проверено управлений: {checked}")
    print(f"лучшее u: {best_a.as_tuple()} | {format_action(best_a, data)}")
    print(f"оценка E[доход]: {best:.4f}")


def main() -> None:
    data = default_problem()
    print("=" * 68)
    print("Лабораторная работа №4")
    print("Пошаговое управление инвестиционным портфелем")
    print("Динамическое программирование / соотношение Беллмана")
    print("=" * 68)
    print("\nНачальное состояние:")
    s0 = data.initial
    print(f"  ЦБ1={s0.v1}, ЦБ2={s0.v2}, Деп={s0.v3}, кэш={s0.cash}, W0={data.w0}")
    print(f"  пакеты={data.packet}, минимумы={data.mins}, комиссии={data.commission}")
    print(f"  квантование состояния: шаг {Q} д.е.")

    print("\n=== Основной результат: критерий математического ожидания ===")
    solver = BellmanSolver(data, "expectation")
    res = solver.solve()
    print(f"Максимальный ожидаемый доход: {res.best_income:.4f} д.е.")
    print(f"Ожидаемое конечное богатство: {res.expected_final_wealth:.4f} д.е.")
    print("Оптимальный план:")
    for i, action in enumerate(res.policy, start=1):
        print(f"  этап {i}: {format_action(action, data)}  k={action.as_tuple()}")

    likely = [max(range(3), key=lambda i: data.stages[t].probs[i]) for t in range(3)]
    path, income = simulate_policy(data, res.policy, likely)
    print("\nСимуляция по наиболее вероятным сценариям:", likely)
    for i, st in enumerate(path):
        print(
            f"  t={i}: ЦБ1={st.v1}, ЦБ2={st.v2}, Деп={st.v3}, кэш={st.cash}, W={st.wealth()}"
        )
    print(f"  доход на этой траектории: {income:.4f}")
    print(f"  состояний в кэше: {len(solver.value_cache)}")

    demo_stage3_fragment(data)

    import sys
    if "--all-criteria" in sys.argv:
        print("\n=== Сравнение критериев ===")
        for name in ("expectation", "wald", "optimistic", "hurwicz"):
            r = BellmanSolver(data, name).solve()
            print(
                f"  {name:12s}: доход={r.best_income:10.4f}  "
                f"план={[a.as_tuple() for a in r.policy]}"
            )


if __name__ == "__main__":
    main()
