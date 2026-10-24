from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable
import random


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


def tiny_problem() -> ProblemData:
    initial = State(to_cents(50), to_cents(200), to_cents(100), to_cents(60))
    stages = [
        StageMarket((0.5, 0.3, 0.2), (1.30, 1.00, 0.70), (1.10, 1.00, 0.90), (1.05, 1.00, 1.00)),
        StageMarket((0.4, 0.4, 0.2), (1.20, 1.00, 0.80), (1.08, 1.00, 0.92), (1.03, 1.00, 1.00)),
    ]
    return ProblemData(
        initial=initial,
        packet=(to_cents(25), to_cents(200), to_cents(100)),
        mins=(to_cents(30), to_cents(150), to_cents(100)),
        commission=(0.04, 0.07, 0.05),
        stages=stages,
        w0=initial.wealth(),
    )


def tiny_problem_nonzero_last() -> ProblemData:
    initial = State(to_cents(100), to_cents(200), to_cents(100), to_cents(0))
    stages = [
        StageMarket((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        StageMarket((1.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
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


def trade_commission_cents(action: Action, data: ProblemData) -> int:
    total = 0
    for i, k in enumerate(action.as_tuple()):
        if k == 0:
            continue
        vol = abs(k) * data.packet[i]
        total += int(round(vol * data.commission[i]))
    return total


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


@dataclass
class MemoryReport:
    n_value: int
    n_policy: int
    n_action_states: int
    n_actions_total: int
    bytes_est: int


def terminal_income(state: State, w0: int) -> float:
    return (state.wealth() - w0) / 100.0


def expected_terminal_after_market(state: State, stage: StageMarket, w0: int) -> float:
    return sum(
        p * terminal_income(apply_market(state, stage, sc), w0)
        for sc, p in enumerate(stage.probs)
    )


def best_last_action_expectation(state: State, data: ProblemData) -> tuple[Action, float]:
    stage = data.stages[-1]
    best_a = Action(0, 0, 0)
    best_v = expected_terminal_after_market(state, stage, data.w0)
    for action in feasible_actions(state, data):
        after = apply_action(state, action, data)
        assert after is not None
        score = expected_terminal_after_market(after, stage, data.w0)
        if score > best_v + 1e-12:
            best_v = score
            best_a = action
    return best_a, best_v


def margins_last_stage_expectation(data: ProblemData) -> list[tuple[float, float]]:
    stage = data.stages[-1]
    er = [
        sum(p * r for p, r in zip(stage.probs, stage.r1)),
        sum(p * r for p, r in zip(stage.probs, stage.r2)),
        sum(p * r for p, r in zip(stage.probs, stage.r3)),
    ]
    out = []
    for i in range(3):
        packet = data.packet[i] / 100.0
        buy = er[i] * packet - packet * (1.0 + data.commission[i])
        sell = packet * (1.0 - data.commission[i]) - er[i] * packet
        out.append((buy, sell))
    return out


def can_accelerate_last_no_trade(data: ProblemData) -> bool:
    if len(data.stages) < 1:
        return False
    for buy, sell in margins_last_stage_expectation(data):
        if buy > 1e-9 or sell > 1e-9:
            return False
    samples = [data.initial]
    rng = random.Random(0)
    for _ in range(40):
        s = State(
            rng.randint(data.mins[0], data.mins[0] + 50_000),
            rng.randint(data.mins[1], data.mins[1] + 200_000),
            rng.randint(data.mins[2], data.mins[2] + 80_000),
            rng.randint(0, 100_000),
        )
        samples.append(s)
    for s in samples:
        a, _ = best_last_action_expectation(s, data)
        if a.as_tuple() != (0, 0, 0):
            return False
    return True


class BellmanSolver:
    def __init__(self, data: ProblemData, criterion_name: str = "expectation"):
        self.data = data
        self.criterion_name = criterion_name
        self.criterion = CRITERIA[criterion_name]
        self.policy_cache: dict[tuple[int, State], Action] = {}
        self.value_cache: dict[tuple[int, State], float] = {}
        self._actions_cache: dict[State, tuple[Action, ...]] = {}
        self.n_stages = len(data.stages)
        self.accelerate_last_no_trade = (
            criterion_name == "expectation" and can_accelerate_last_no_trade(data)
        )
        self.acceleration_checks = 0
        self.acceleration_failures = 0

    def _actions(self, state: State) -> tuple[Action, ...]:
        cached = self._actions_cache.get(state)
        if cached is None:
            cached = tuple(feasible_actions(state, self.data))
            self._actions_cache[state] = cached
        return cached

    def value(self, stage_idx: int, state: State) -> float:
        key = (stage_idx, state)
        hit = self.value_cache.get(key)
        if hit is not None:
            return hit

        if stage_idx >= self.n_stages:
            val = terminal_income(state, self.data.w0)
            self.value_cache[key] = val
            return val

        stage = self.data.stages[stage_idx]
        last = stage_idx == self.n_stages - 1
        best_val = float("-inf")
        best_action: Action | None = None

        if last and self.criterion_name == "expectation":
            best_action, best_val = best_last_action_expectation(state, self.data)
            if self.accelerate_last_no_trade:
                self.acceleration_checks += 1
                if best_action.as_tuple() != (0, 0, 0):
                    self.acceleration_failures += 1
                    self.accelerate_last_no_trade = False
            self.policy_cache[key] = best_action
            self.value_cache[key] = best_val
            return best_val

        for action in self._actions(state):
            after = apply_action(state, action, self.data)
            assert after is not None
            if (
                self.criterion_name == "expectation"
                and self.accelerate_last_no_trade
                and stage_idx == self.n_stages - 2
            ):
                scenario_vals = [
                    expected_terminal_after_market(
                        apply_market(after, stage, sc),
                        self.data.stages[-1],
                        self.data.w0,
                    )
                    for sc in range(3)
                ]
                score = criterion_expectation(scenario_vals, stage.probs)
            else:
                scenario_vals = [
                    self.value(stage_idx + 1, apply_market(after, stage, sc))
                    for sc in range(3)
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

    def memory_report(self) -> MemoryReport:
        n_actions_total = sum(len(v) for v in self._actions_cache.values())
        bytes_est = (
            len(self.value_cache) * 80
            + len(self.policy_cache) * 96
            + len(self._actions_cache) * 64
            + n_actions_total * 48
        )
        return MemoryReport(
            n_value=len(self.value_cache),
            n_policy=len(self.policy_cache),
            n_action_states=len(self._actions_cache),
            n_actions_total=n_actions_total,
            bytes_est=bytes_est,
        )

    def solve(self) -> SolveResult:
        start = self.data.initial
        best = self.value(0, start)
        policy_by_state = self._collect_policy_tree(start)

        if self.criterion_name == "expectation" and len(policy_by_state) >= 1:
            for st, act in policy_by_state[-1].items():
                a2, _ = best_last_action_expectation(st, self.data)
                if a2.as_tuple() != act.as_tuple():
                    raise RuntimeError("policy last stage mismatch with direct check")

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
    assert abs(solver.value(len(data.stages), s0) - 0.0) < 1e-9
    print("T5 V_terminal(s0)=0: OK")
    ok += 1

    assert apply_action(s0, Action(-3, 0, 0), data) is None
    print("T6 минимум ЦБ1: OK")
    ok += 1

    tiny = tiny_problem()
    s_tiny = BellmanSolver(tiny, "expectation")
    v_dp = s_tiny.solve().best_income

    def full_tree_best(stage_idx: int, state: State) -> float:
        if stage_idx >= len(tiny.stages):
            return terminal_income(state, tiny.w0)
        best = float("-inf")
        stage = tiny.stages[stage_idx]
        for action in feasible_actions(state, tiny):
            after = apply_action(state, action, tiny)
            assert after is not None
            vals = [
                full_tree_best(stage_idx + 1, apply_market(after, stage, sc))
                for sc in range(3)
            ]
            score = criterion_expectation(vals, stage.probs)
            if score > best:
                best = score
        return best

    v_bf = full_tree_best(0, tiny.initial)
    assert abs(v_dp - v_bf) < 1e-6
    print(f"T7 tiny DP == полный перебор: {v_dp:.4f}: OK")
    ok += 1

    nz = tiny_problem_nonzero_last()
    assert can_accelerate_last_no_trade(nz) is False
    s_nz = BellmanSolver(nz, "expectation")
    assert s_nz.accelerate_last_no_trade is False
    r_nz = s_nz.solve()
    last_nonzero = any(a.as_tuple() != (0, 0, 0) for a in r_nz.policy_by_state[-1].values())
    assert last_nonzero
    print("T8 last-stage ненулевое u + без ускорения: OK")
    ok += 1

    assert can_accelerate_last_no_trade(data) is True
    print("T9 условие ускорения для исходных данных: OK")
    ok += 1

    print(f"Пройдено тестов: {ok}/9")


def estimate_complexity(solver: BellmanSolver, n_actions_s0: int) -> None:
    mem = solver.memory_report()
    print("\n=== Оценка сложности и памяти ===")
    print(f"K={solver.n_stages}, |Ξ|=3, |A(s0)|={n_actions_s0}")
    print(f"value_cache: {mem.n_value} ключей")
    print(f"policy_cache: {mem.n_policy} ключей")
    print(
        f"_actions_cache: {mem.n_action_states} состояний, "
        f"всего закэшированных действий {mem.n_actions_total}"
    )
    print(
        f"Оценка памяти ≈ {mem.bytes_est} байт (~{mem.bytes_est/1024:.1f} КиБ): "
        f"values + policy + actions"
    )
    print("Время O(K·|S|·|A|·|Ξ|)")
    print(
        f"Ускорение last-no-trade: {solver.accelerate_last_no_trade}, "
        f"проверок={solver.acceleration_checks}, сбоев={solver.acceleration_failures}"
    )


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
    print(
        "Алгоритм: рекурсия Беллмана с мемоизацией; на последнем этапе полный перебор u; "
        "ускорение E[V4(market(s))] только если доказано и проверяется, что u*=(0,0,0)."
    )

    run_tests(data)

    n_a0 = len(feasible_actions(data.initial, data))
    print(f"\n|A(s0)| = {n_a0}")

    print("\n=== Критерий: математическое ожидание ===")
    t0 = time.time()
    solver = BellmanSolver(data, "expectation")
    res = solver.solve()
    print(f"Время счёта: {time.time() - t0:.1f} с")
    print(f"I* = {res.best_income:.4f} д.е.")
    print(f"E[W_final] = {res.expected_final_wealth:.4f} д.е.")

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

    estimate_complexity(solver, n_a0)

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
