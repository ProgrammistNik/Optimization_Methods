from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable


def to_cents(x: float) -> int:
    return int(round(float(x) * 100.0 + 1e-9))


def rub(cents: int) -> float:
    return cents / 100.0


@dataclass(frozen=True)
class State:
    v1: int
    v2: int
    v3: int
    cash: int

    def wealth(self) -> int:
        return self.v1 + self.v2 + self.v3 + self.cash

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.v1, self.v2, self.v3, self.cash


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
    initial = State(to_cents(100), to_cents(800), to_cents(400), to_cents(600))
    stages = [
        StageMarket((0.6, 0.3, 0.1), (1.20, 1.05, 0.80), (1.10, 1.02, 0.95), (1.07, 1.03, 1.00)),
        StageMarket((0.3, 0.2, 0.5), (1.40, 1.05, 0.60), (1.15, 1.00, 0.90), (1.01, 1.00, 1.00)),
        StageMarket((0.4, 0.4, 0.2), (1.15, 1.05, 0.70), (1.12, 1.01, 0.94), (1.05, 1.01, 1.00)),
    ]
    return ProblemData(
        initial=initial,
        packet=(to_cents(25), to_cents(200), to_cents(100)),
        mins=(to_cents(30), to_cents(150), to_cents(100)),
        commission=(0.04, 0.07, 0.05),
        stages=stages,
        w0=initial.wealth(),
    )


def apply_action(state: State, action: Action, data: ProblemData) -> State | None:
    vals = [state.v1, state.v2, state.v3]
    ks = action.as_tuple()
    cash = state.cash

    for i in range(3):
        if ks[i] < 0:
            vol = (-ks[i]) * data.packet[i]
            cash += int(round(vol * (1.0 - data.commission[i])))
            vals[i] -= vol

    for i in range(3):
        if ks[i] > 0:
            vol = ks[i] * data.packet[i]
            cash -= int(round(vol * (1.0 + data.commission[i])))
            vals[i] += vol

    if cash < 0:
        return None
    for i in range(3):
        if vals[i] < 0:
            return None
        if ks[i] < 0 and vals[i] < data.mins[i]:
            return None
    return State(vals[0], vals[1], vals[2], cash)


def apply_market(state: State, stage: StageMarket, scenario: int) -> State:
    return State(
        int(round(state.v1 * stage.r1[scenario])),
        int(round(state.v2 * stage.r2[scenario])),
        int(round(state.v3 * stage.r3[scenario])),
        state.cash,
    )


def feasible_actions(state: State, data: ProblemData) -> list[Action]:
    sell_max = [
        max(int((v - data.mins[i]) // data.packet[i]), 0)
        for i, v in enumerate((state.v1, state.v2, state.v3))
    ]
    unit = [int(round(data.packet[i] * (1.0 + data.commission[i]))) for i in range(3)]
    sell_unit = [int(round(data.packet[i] * (1.0 - data.commission[i]))) for i in range(3)]
    actions: list[Action] = []
    seen: set[tuple[int, int, int]] = set()

    for s1, s2, s3 in product(
        range(sell_max[0] + 1),
        range(sell_max[1] + 1),
        range(sell_max[2] + 1),
    ):
        vals = [state.v1, state.v2, state.v3]
        cash = state.cash
        ok = True
        for i, s in enumerate((s1, s2, s3)):
            if s == 0:
                continue
            vol = s * data.packet[i]
            if vals[i] - vol < data.mins[i]:
                ok = False
                break
            cash += s * sell_unit[i]
            vals[i] -= vol
        if not ok:
            continue

        max_b1 = cash // unit[0] if unit[0] else 0
        for b1 in range(max_b1 + 1):
            rem1 = cash - b1 * unit[0]
            max_b2 = rem1 // unit[1] if unit[1] else 0
            for b2 in range(max_b2 + 1):
                rem2 = rem1 - b2 * unit[1]
                max_b3 = rem2 // unit[2] if unit[2] else 0
                for b3 in range(max_b3 + 1):
                    key = (b1 - s1, b2 - s2, b3 - s3)
                    if key in seen:
                        continue
                    seen.add(key)
                    action = Action(*key)
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
    policy_by_state: list[dict[State, Action]]


class BellmanSolver:
    def __init__(self, data: ProblemData, criterion_name: str = "expectation"):
        self.data = data
        self.criterion_name = criterion_name
        self.criterion = CRITERIA[criterion_name]
        self.policy_cache: dict[tuple[int, State], Action] = {}
        self.value_cache: dict[tuple[int, State], float] = {}
        self._actions_cache: dict[State, tuple[Action, ...]] = {}
        self.n_stages = len(data.stages)
        st = data.stages[2]
        self._er3 = (
            sum(p * r for p, r in zip(st.probs, st.r1)),
            sum(p * r for p, r in zip(st.probs, st.r2)),
            sum(p * r for p, r in zip(st.probs, st.r3)),
        )

    def _actions(self, state: State) -> tuple[Action, ...]:
        cached = self._actions_cache.get(state)
        if cached is None:
            cached = tuple(feasible_actions(state, self.data))
            self._actions_cache[state] = cached
        return cached

    def _terminal_income(self, state: State) -> float:
        return (state.wealth() - self.data.w0) / 100.0

    def _expected_v3_no_trade(self, state: State) -> float:
        e1, e2, e3 = self._er3
        return (e1 * state.v1 + e2 * state.v2 + e3 * state.v3 + state.cash - self.data.w0) / 100.0

    def value(self, stage_idx: int, state: State) -> float:
        key = (stage_idx, state)
        hit = self.value_cache.get(key)
        if hit is not None:
            return hit

        if stage_idx >= self.n_stages:
            val = self._terminal_income(state)
            self.value_cache[key] = val
            return val

        if self.criterion_name == "expectation" and stage_idx == 2:
            best_action = Action(0, 0, 0)
            best_val = self._expected_v3_no_trade(state)
            for action in self._actions(state):
                after = apply_action(state, action, self.data)
                assert after is not None
                score = self._expected_v3_no_trade(after)
                if score > best_val + 1e-12:
                    best_val = score
                    best_action = action
            self.policy_cache[key] = best_action
            self.value_cache[key] = best_val
            return best_val

        stage = self.data.stages[stage_idx]
        best_val = float("-inf")
        best_action: Action | None = None

        for action in self._actions(state):
            after = apply_action(state, action, self.data)
            assert after is not None
            if self.criterion_name == "expectation" and stage_idx == 1:
                score = criterion_expectation(
                    [self._expected_v3_no_trade(apply_market(after, stage, sc)) for sc in range(3)],
                    stage.probs,
                )
            elif stage_idx == self.n_stages - 1:
                scenario_vals = [
                    self._terminal_income(apply_market(after, stage, sc)) for sc in range(3)
                ]
                score = self.criterion(scenario_vals, stage.probs)
            else:
                scenario_vals = [
                    self.value(stage_idx + 1, apply_market(after, stage, sc)) for sc in range(3)
                ]
                score = self.criterion(scenario_vals, stage.probs)
            if score > best_val:
                best_val = score
                best_action = action

        if best_action is not None:
            self.policy_cache[key] = best_action
        self.value_cache[key] = best_val
        return best_val

    def _collect_policy_tree(self, start: State) -> list[dict[State, Action]]:
        by_stage: list[dict[State, Action]] = [{} for _ in self.data.stages]
        frontier = {start}
        for stage_idx, stage in enumerate(self.data.stages):
            nxt: set[State] = set()
            for state in frontier:
                if (stage_idx, state) not in self.policy_cache:
                    self.value(stage_idx, state)
                action = self.policy_cache[(stage_idx, state)]
                by_stage[stage_idx][state] = action
                after = apply_action(state, action, self.data)
                assert after is not None
                for sc in range(3):
                    nxt.add(apply_market(after, stage, sc))
            frontier = nxt
        return by_stage

    def solve(self) -> SolveResult:
        start = self.data.initial
        best = self.value(0, start)
        policy_by_state = self._collect_policy_tree(start)

        policy: list[Action] = []
        path = [start]
        state = start
        for stage_idx in range(self.n_stages):
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
            expected_final_wealth=best + self.data.w0 / 100.0,
            policy=policy,
            path_states=path,
            policy_by_state=policy_by_state,
        )


def format_action(action: Action, data: ProblemData) -> str:
    names = ("ЦБ1", "ЦБ2", "Деп")
    parts = []
    for name, k, d in zip(names, action.as_tuple(), data.packet):
        if k == 0:
            continue
        op = "купить" if k > 0 else "продать"
        parts.append(f"{op} {name} на {rub(abs(k) * d):g}")
    return "; ".join(parts) if parts else "без сделок"


def format_state(state: State) -> str:
    return (
        f"(ЦБ1={rub(state.v1):g}, ЦБ2={rub(state.v2):g}, Деп={rub(state.v3):g}, "
        f"кэш={rub(state.cash):g}, W={rub(state.wealth()):g})"
    )


def run_tests(data: ProblemData) -> None:
    print("\n=== Независимые тесты ===")
    ok = 0
    s0 = data.initial

    after = apply_action(s0, Action(-1, 0, 0), data)
    assert after is not None
    assert after.cash == s0.cash + int(round(data.packet[0] * 0.96))
    assert after.v1 == s0.v1 - data.packet[0]
    print("T1 продажа ЦБ1: OK")
    ok += 1

    after2 = apply_action(s0, Action(-2, 3, 0), data)
    assert after2 is not None
    assert after2.v2 == s0.v2 + 3 * data.packet[1]
    print("T2 покупка за счёт продаж: OK")
    ok += 1

    assert apply_action(s0, Action(0, 3, 0), data) is None
    print("T3 запрет кредита: OK")
    ok += 1

    acts = feasible_actions(s0, data)
    assert Action(-2, 3, 0) in acts
    assert Action(0, 3, 0) not in acts
    print("T4 генерация действий с учётом продаж: OK")
    ok += 1

    solver = BellmanSolver(data, "expectation")
    assert abs(solver.value(3, s0) - 0.0) < 1e-9
    print("T5 V4(s0)=0: OK")
    ok += 1

    assert apply_action(s0, Action(-3, 0, 0), data) is None
    print("T6 минимум ЦБ1: OK")
    ok += 1

    print(f"Пройдено тестов: {ok}/6")


def demo_manual_bellman(data: ProblemData) -> None:
    print("\n=== Ручной фрагмент Беллмана (этап 3) ===")
    s = data.initial
    stage = data.stages[2]
    print("s = (100, 800, 400, 600), W0 = 1900")
    print("V4(s') = W(s') - W0")
    print("V3(s) = max_u Σ_ξ p3(ξ)·V4(f(s,u,ξ))")
    candidates = [
        Action(0, 0, 0),
        Action(-1, 0, 0),
        Action(-2, 0, 0),
        Action(0, 1, 0),
        Action(-1, 1, 0),
    ]
    best = float("-inf")
    best_a = None
    for action in candidates:
        after = apply_action(s, action, data)
        if after is None:
            print(f"  u={action.as_tuple()}: недопустимо")
            continue
        vals = [
            (apply_market(after, stage, sc).wealth() - data.w0) / 100.0 for sc in range(3)
        ]
        score = criterion_expectation(vals, stage.probs)
        print(f"  u={action.as_tuple()}: {[round(v, 2) for v in vals]}, E={score:.4f}")
        if score > best:
            best, best_a = score, action
    print(f"Лучшее в фрагменте: u={best_a.as_tuple()}, E={best:.4f}")


def estimate_complexity(n_actions: int, n_states: int, n_stages: int = 3) -> None:
    print("\n=== Оценка сложности ===")
    print(f"K={n_stages}, |Ξ|=3, |A(s0)|={n_actions}, |S|≈{n_states}")
    print("Время O(K·|S|·|A|·|Ξ|), память O(|S|)")


def main() -> None:
    import sys
    import time

    data = default_problem()
    print("=" * 68)
    print("Лабораторная работа №4")
    print("Пошаговое управление инвестиционным портфелем")
    print("=" * 68)
    print("\nНачальное состояние:")
    print(f"  {format_state(data.initial)}, W0={rub(data.w0):g}")
    print(
        f"  пакеты={[rub(x) for x in data.packet]}, "
        f"минимумы={[rub(x) for x in data.mins]}, комиссии={list(data.commission)}"
    )

    run_tests(data)
    demo_manual_bellman(data)

    n_a0 = len(feasible_actions(data.initial, data))
    print(f"\n|A(s0)| = {n_a0}")

    print("\n=== Критерий: математическое ожидание ===")
    t0 = time.time()
    solver = BellmanSolver(data, "expectation")
    res = solver.solve()
    print(f"Время счёта: {time.time() - t0:.1f} с")
    print(f"I* = {res.best_income:.4f} д.е.")
    print(f"E[W_final] = {res.expected_final_wealth:.4f} д.е.")
    print(f"Состояний в кэше: {len(solver.value_cache)}")

    print("\n--- Политика A_k(s) для достижимых состояний ---")
    for k, mapping in enumerate(res.policy_by_state, start=1):
        print(f"Этап {k}: {len(mapping)} состояний")
        items = sorted(mapping.items(), key=lambda kv: kv[0].as_tuple())
        for i, (st, act) in enumerate(items):
            if i >= 60:
                print(f"  ... ещё {len(items) - 60}")
                break
            print(f"  s={format_state(st)}")
            print(f"    → u={act.as_tuple()} | {format_action(act, data)}")

    print("\n--- Траектория при наиболее вероятных сценариях ---")
    for i, action in enumerate(res.policy, start=1):
        print(f"  этап {i}: {format_action(action, data)}  k={action.as_tuple()}")
    for i, st in enumerate(res.path_states):
        print(f"  t={i}: {format_state(st)}")

    estimate_complexity(n_a0, len(solver.value_cache))

    if "--all-criteria" in sys.argv:
        print("\n=== Сравнение критериев ===")
        for name in ("expectation", "wald", "optimistic", "hurwicz"):
            t1 = time.time()
            r = BellmanSolver(data, name).solve()
            print(
                f"  {name:12s}: I*={r.best_income:10.4f}  "
                f"план={[a.as_tuple() for a in r.policy]}  ({time.time() - t1:.1f}с)"
            )


if __name__ == "__main__":
    main()
