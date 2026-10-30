from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable
import random
import tracemalloc


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


def identity_one_stage(cash: int = 600) -> ProblemData:
    """Identity market: all r=1 — оптимально не торговать (комиссии)."""
    initial = State(to_cents(100), to_cents(800), to_cents(400), to_cents(cash))
    stages = [
        StageMarket((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
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
        # Минимум проверяется ТОЛЬКО при продаже; после рынка vi может быть < mins[i].
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
    """Полный перебор на последнем этапе. Tie-break: первый с лучшим score (строгое >)."""
    stage = data.stages[-1]
    best_a = Action(0, 0, 0)
    best_v = expected_terminal_after_market(state, stage, data.w0)
    for action in feasible_actions(state, data):
        after = apply_action(state, action, data)
        assert after is not None
        score = expected_terminal_after_market(after, stage, data.w0)
        # Строгое >: при равенстве оставляем ранее найденное (первое) действие.
        if score > best_v:
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
            # Tie-break: строгое > — при равенстве сохраняем первое найденное u.
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

    def score_action_at_stage0(self, action: Action) -> float | None:
        """Оценка u в s0 через уже заполненный value_cache (после solve)."""
        after = apply_action(self.data.initial, action, self.data)
        if after is None:
            return None
        stage = self.data.stages[0]
        vals = []
        for sc in range(3):
            nxt = apply_market(after, stage, sc)
            key = (1, nxt)
            if key not in self.value_cache:
                vals.append(self.value(1, nxt))
            else:
                vals.append(self.value_cache[key])
        return criterion_expectation(vals, stage.probs)

    def uniqueness_at_s0(self) -> tuple[float, list[Action], int]:
        """Среди всех допустимых A(s0): максимум и число оптимумов. Переиспользует кэши."""
        start = self.data.initial
        acts = self._actions(start)
        best_val = float("-inf")
        best_actions: list[Action] = []
        for action in acts:
            score = self.score_action_at_stage0(action)
            assert score is not None
            if score > best_val:
                best_val = score
                best_actions = [action]
            elif abs(score - best_val) <= 1e-12:
                best_actions.append(action)
        return best_val, best_actions, len(acts)

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


def _print_test(tid: str, title: str, inp: str, expected: str, actual: str, ok: bool) -> None:
    status = "OK" if ok else "FAIL"
    print(f"{tid} {title}: {status}")
    print(f"  Вход:      {inp}")
    print(f"  Ожидаемо:  {expected}")
    print(f"  Факт:      {actual}")
    if not ok:
        raise AssertionError(f"{tid} failed")


def run_tests(data: ProblemData) -> None:
    print("\n=== Независимые тесты ===")
    ok_n = 0
    s0 = data.initial

    after = apply_action(s0, Action(-1, 0, 0), data)
    exp_cash = s0.cash + int(round(data.packet[0] * 0.96))
    exp_v1 = s0.v1 - data.packet[0]
    actual = f"cash={after.cash if after else None}, v1={after.v1 if after else None}"
    _print_test(
        "T1", "продажа ЦБ1", "s0, u=(-1,0,0)",
        f"cash={exp_cash}, v1={exp_v1}", actual,
        after is not None and after.cash == exp_cash and after.v1 == exp_v1,
    )
    ok_n += 1

    after2 = apply_action(s0, Action(-2, 3, 0), data)
    exp_v2 = s0.v2 + 3 * data.packet[1]
    _print_test(
        "T2", "покупка за счёт продаж", "s0, u=(-2,3,0)",
        f"v2={exp_v2}", f"v2={after2.v2 if after2 else None}",
        after2 is not None and after2.v2 == exp_v2,
    )
    ok_n += 1

    bad = apply_action(s0, Action(0, 3, 0), data)
    _print_test(
        "T3", "запрет кредита", "s0, u=(0,3,0) без продаж",
        "None (cash<0)", str(bad), bad is None,
    )
    ok_n += 1

    acts = feasible_actions(s0, data)
    has_ok = Action(-2, 3, 0) in acts
    has_bad = Action(0, 3, 0) in acts
    _print_test(
        "T4", "генерация действий", "A(s0)",
        "(-2,3,0) in A; (0,3,0) not in A",
        f"(-2,3,0)={has_ok}; (0,3,0)={has_bad}",
        has_ok and not has_bad,
    )
    ok_n += 1

    solver = BellmanSolver(data, "expectation")
    v_term = solver.value(len(data.stages), s0)
    _print_test(
        "T5", "терминал V(s0)=0", f"stage={len(data.stages)}, s0",
        "0.0", f"{v_term}", abs(v_term - 0.0) < 1e-9,
    )
    ok_n += 1

    below_min = apply_action(s0, Action(-3, 0, 0), data)
    _print_test(
        "T6", "минимум ЦБ1 при продаже", "s0, u=(-3,0,0) → v1=25<30",
        "None", str(below_min), below_min is None,
    )
    ok_n += 1

    tiny = tiny_problem()
    s_tiny = BellmanSolver(tiny, "expectation")
    v_dp = s_tiny.solve().best_income

    def full_tree_best(stage_idx: int, state: State) -> float:
        if stage_idx >= len(tiny.stages):
            return terminal_income(state, tiny.w0)
        best = float("-inf")
        stage = tiny.stages[stage_idx]
        for action in feasible_actions(state, tiny):
            after_t = apply_action(state, action, tiny)
            assert after_t is not None
            vals = [
                full_tree_best(stage_idx + 1, apply_market(after_t, stage, sc))
                for sc in range(3)
            ]
            score = criterion_expectation(vals, stage.probs)
            if score > best:
                best = score
        return best

    v_bf = full_tree_best(0, tiny.initial)
    _print_test(
        "T7", "tiny DP == полный перебор", "tiny_problem K=2",
        "DP==BF, |diff|<1e-6",
        f"DP={v_dp:.6f}, BF={v_bf:.6f}, diff={abs(v_dp - v_bf):.2e}",
        abs(v_dp - v_bf) < 1e-6,
    )
    ok_n += 1

    nz = tiny_problem_nonzero_last()
    assert can_accelerate_last_no_trade(nz) is False
    s_nz = BellmanSolver(nz, "expectation")
    assert s_nz.accelerate_last_no_trade is False
    r_nz = s_nz.solve()
    last_nonzero = any(a.as_tuple() != (0, 0, 0) for a in r_nz.policy_by_state[-1].values())
    _print_test(
        "T8", "last-stage ненулевое u", "tiny_problem_nonzero_last",
        "accelerate=False; exists u≠0 на last",
        f"accelerate={s_nz.accelerate_last_no_trade}; nonzero={last_nonzero}",
        (not s_nz.accelerate_last_no_trade) and last_nonzero,
    )
    ok_n += 1

    acc = can_accelerate_last_no_trade(data)
    _print_test(
        "T9", "условие ускорения исходных данных",
        "default_problem margins buy/sell all <0 + sample check",
        "True", str(acc), acc is True,
    )
    ok_n += 1

    print("\n=== Тесты по методичке §8.3 (адаптация к портфелю) ===")

    z = State(s0.v1, s0.v2, s0.v3, 0)
    z_acts = feasible_actions(z, data)
    buys_only = [a for a in z_acts if any(k > 0 for k in a.as_tuple()) and all(k >= 0 for k in a.as_tuple())]
    _print_test(
        "T8_3_1", "нулевой кэш → нет чистых покупок",
        "s=(v1,v2,v3,cash=0)",
        "|buys_only|=0; (0,0,0) допустимо",
        f"|A|={len(z_acts)}; buys_only={len(buys_only)}; zero={(Action(0,0,0) in z_acts)}",
        len(buys_only) == 0 and Action(0, 0, 0) in z_acts,
    )
    ok_n += 1

    idp = identity_one_stage()
    r_id = BellmanSolver(idp, "expectation").solve()
    _print_test(
        "T8_3_2", "тождественный рынок → без сделок",
        "1 этап, все r=1", "u*=(0,0,0); I*=0",
        f"u={r_id.policy[0].as_tuple()}; I*={r_id.best_income:.6f}",
        r_id.policy[0].as_tuple() == (0, 0, 0) and abs(r_id.best_income) < 1e-9,
    )
    ok_n += 1

    tiny_u = tiny_problem()
    st = BellmanSolver(tiny_u, "expectation")
    st.solve()
    scores = []
    for a in feasible_actions(tiny_u.initial, tiny_u):
        after_a = apply_action(tiny_u.initial, a, tiny_u)
        assert after_a is not None
        stage = tiny_u.stages[0]
        vals = [st.value(1, apply_market(after_a, stage, sc)) for sc in range(3)]
        scores.append((a.as_tuple(), criterion_expectation(vals, stage.probs)))
    best_sc = max(s for _, s in scores)
    optima = [u for u, s in scores if abs(s - best_sc) < 1e-9]
    _print_test(
        "T8_3_3", "единственность оптимума (tiny)",
        "tiny A(s0)", "ровно один оптимум u=(2,0,0)",
        f"optima={optima}, V*={best_sc:.6f}",
        len(optima) == 1 and optima[0] == (2, 0, 0),
    )
    ok_n += 1

    unit0 = int(round(data.packet[0] * (1.0 + data.commission[0])))
    max_k1 = s0.cash // unit0
    after_all = apply_action(s0, Action(max_k1, 0, 0), data)
    _print_test(
        "T8_3_4", "весь кэш в один актив (ЦБ1)",
        f"u=({max_k1},0,0), unit={rub(unit0)}",
        f"допустимо; cash_остаток < unit; v1 увеличен на {max_k1}*25",
        f"after={format_state(after_all) if after_all else None}; cash_left={after_all.cash if after_all else None}",
        after_all is not None
        and after_all.cash < unit0
        and after_all.v1 == s0.v1 + max_k1 * data.packet[0],
    )
    ok_n += 1

    inv_dep = apply_action(s0, Action(0, 0, -4), data)
    inv_neg_cash_buy = apply_action(s0, Action(100, 0, 0), data)
    _print_test(
        "T8_3_5", "недопустимые входы",
        "u=(0,0,-4) ниже min; u=(100,0,0) нехватка кэша",
        "оба None",
        f"(0,0,-4)={inv_dep}; (100,0,0)={inv_neg_cash_buy}",
        inv_dep is None and inv_neg_cash_buy is None,
    )
    ok_n += 1

    s_low = State(to_cents(35), to_cents(880), to_cents(428), to_cents(746))
    stage2 = data.stages[1]
    after_neg = apply_market(s_low, stage2, 2)
    _print_test(
        "T8_3_6", "после рынка ЦБ1<30 допустимо",
        f"s={format_state(s_low)}, ξ=негатив этапа2",
        "ЦБ1=21 (< min=30), состояние допустимо",
        f"ЦБ1={rub(after_neg.v1)}",
        after_neg.v1 == to_cents(21),
    )
    ok_n += 1

    print(f"Пройдено тестов: {ok_n}")


def estimate_complexity(
    solver: BellmanSolver,
    n_actions_s0: int,
    tracemalloc_peak: int | None = None,
) -> None:
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
        f"Оценка памяти (формула): ≈ {mem.bytes_est} байт "
        f"(~{mem.bytes_est / (1024 * 1024):.1f} МиБ): "
        f"80·n_value + 96·n_policy + 64·n_action_states + 48·n_actions_total"
    )
    if tracemalloc_peak is not None:
        print(
            f"Измерено tracemalloc peak: {tracemalloc_peak} байт "
            f"(~{tracemalloc_peak / (1024 * 1024):.1f} МиБ)"
        )
    print("Время O(K·|S|·|A|·|Ξ|)")
    margins = margins_last_stage_expectation(solver.data)
    names = ("ЦБ1", "ЦБ2", "Деп")
    print("Маржинальные приросты last-stage (buy, sell) при E[r]:")
    for name, (b, s) in zip(names, margins):
        print(f"  {name}: buy={b:.4f}, sell={s:.4f}")
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
    print("Tie-break: if score > best_val — при равенстве сохраняется первое u.")

    run_tests(data)

    n_a0 = len(feasible_actions(data.initial, data))
    print(f"\n|A(s0)| = {n_a0}")

    print("\n=== Критерий: математическое ожидание ===")
    t0 = time.time()
    solver = BellmanSolver(data, "expectation")
    print(f"accelerate_last_no_trade={solver.accelerate_last_no_trade}")
    tracemalloc.start()
    res = solver.solve()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Время счёта: {time.time() - t0:.1f} с")
    print(f"I* = {res.best_income:.5f} д.е.  (≈ {res.best_income:.2f})")
    print(f"E[W_final] = {res.expected_final_wealth:.4f} д.е.")

    best_u, optima, n_acts = solver.uniqueness_at_s0()
    print("\n--- Единственность A1(s0) среди всех допустимых ---")
    print(f"|A(s0)|={n_acts}, V*={best_u:.5f}")
    print(f"число оптимумов={len(optima)}: {[a.as_tuple() for a in optima]}")
    if len(optima) == 1:
        print(f"u1*={optima[0].as_tuple()} — ЕДИНСТВЕННЫЙ максимум")
    for neigh in (Action(22, 0, 0), Action(23, 0, 0), Action(24, 0, 0), Action(0, 0, 0)):
        sc = solver.score_action_at_stage0(neigh)
        print(f"  сосед {neigh.as_tuple()}: {'недопустим' if sc is None else f'{sc:.5f}'}")

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

    estimate_complexity(solver, n_a0, tracemalloc_peak=peak)

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
