from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_dp import (
    Action,
    BellmanSolver,
    ProblemData,
    State,
    apply_action,
    apply_market,
    default_problem,
    format_action,
    format_state,
    rub,
    trade_commission_cents,
)

SCENARIO_NAMES = ("благопр.", "нейтр.", "негатив.")


@dataclass
class Leaf:
    scenarios: tuple[int, ...]
    probability: float
    actions: list[Action]
    commissions: list[float]
    final_wealth: float
    income: float
    return_rate: float
    path_states: list[State]


def build_outcome_tree(data: ProblemData, solver: BellmanSolver) -> list[Leaf]:
    leaves: list[Leaf] = []

    def dfs(
        stage_idx: int,
        state: State,
        scenarios: list[int],
        prob: float,
        actions: list[Action],
        commissions: list[float],
        path: list[State],
    ) -> None:
        if stage_idx >= len(data.stages):
            w = rub(state.wealth())
            income = w - rub(data.w0)
            leaves.append(
                Leaf(
                    scenarios=tuple(scenarios),
                    probability=prob,
                    actions=list(actions),
                    commissions=list(commissions),
                    final_wealth=w,
                    income=income,
                    return_rate=income / rub(data.w0),
                    path_states=list(path),
                )
            )
            return

        key = (stage_idx, state)
        if key not in solver.policy_cache:
            solver.value(stage_idx, state)
        action = solver.policy_cache[key]
        after = apply_action(state, action, data)
        assert after is not None
        stage = data.stages[stage_idx]
        fee = rub(trade_commission_cents(action, data))
        for sc in range(3):
            if stage.probs[sc] <= 0:
                continue
            nxt = apply_market(after, stage, sc)
            dfs(
                stage_idx + 1,
                nxt,
                scenarios + [sc],
                prob * stage.probs[sc],
                actions + [action],
                commissions + [fee],
                path + [nxt],
            )

    dfs(0, data.initial, [], 1.0, [], [], [data.initial])
    return leaves


def summarize(leaves: list[Leaf]) -> dict:
    return {
        "n_leaves": len(leaves),
        "E_W": sum(L.probability * L.final_wealth for L in leaves),
        "E_income": sum(L.probability * L.income for L in leaves),
        "E_return": sum(L.probability * L.return_rate for L in leaves),
        "P_loss": sum(L.probability for L in leaves if L.income < 0),
        "min_income": min(L.income for L in leaves),
        "max_income": max(L.income for L in leaves),
    }


def write_tree_txt(
    data: ProblemData, solver: BellmanSolver, leaves: list[Leaf], out_path: Path
) -> None:
    summary = summarize(leaves)
    text = []
    text.append("ДЕРЕВО ИСХОДОВ (оптимальная политика, критерий ожидания)")
    text.append(f"s0 = {format_state(data.initial)}")
    text.append(f"листьев = {summary['n_leaves']}")
    text.append("")

    def print_node(stage_idx: int, state: State, prefix: str, prob: float) -> None:
        if stage_idx >= len(data.stages):
            income = rub(state.wealth()) - rub(data.w0)
            text.append(
                f"{prefix}ЛИСТ W={rub(state.wealth()):g}, доход={income:.2f}, P={prob:.6f}"
            )
            return
        action = solver.policy_cache[(stage_idx, state)]
        text.append(
            f"{prefix}этап {stage_idx+1}: s={format_state(state)} | "
            f"u={action.as_tuple()} {format_action(action, data)}"
        )
        after = apply_action(state, action, data)
        assert after is not None
        stage = data.stages[stage_idx]
        for sc in range(3):
            if stage.probs[sc] <= 0:
                continue
            nxt = apply_market(after, stage, sc)
            text.append(f"{prefix}  └─ {SCENARIO_NAMES[sc]} p={stage.probs[sc]:.2f}")
            print_node(stage_idx + 1, nxt, prefix + "    ", prob * stage.probs[sc])

    print_node(0, data.initial, "", 1.0)
    text.append("")
    text.append("ЛИСТЬЯ (подробно)")
    for i, L in enumerate(leaves, start=1):
        seq = " -> ".join(SCENARIO_NAMES[s] for s in L.scenarios)
        ops = " | ".join(format_action(a, data) for a in L.actions)
        fees = sum(L.commissions)
        text.append(
            f"{i}. {seq}; P={L.probability:.6f}; ops: {ops}; "
            f"комиссии={fees:.2f}; W={L.final_wealth:.2f}; "
            f"I={L.income:.2f}; r={L.return_rate*100:.2f}%"
        )
    text.append("")
    text.append(
        f"E[W]={summary['E_W']:.4f}, E[I]={summary['E_income']:.4f}, "
        f"E[r]={summary['E_return']*100:.4f}%, P(loss)={summary['P_loss']:.4f}, "
        f"minI={summary['min_income']:.4f}, maxI={summary['max_income']:.4f}"
    )
    out_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    data = default_problem()
    solver = BellmanSolver(data, "expectation")
    print("Считаю оптимальную политику...")
    res = solver.solve()
    print(f"I* = {res.best_income:.4f}")
    leaves = build_outcome_tree(data, solver)
    summary = summarize(leaves)
    out = Path(__file__).resolve().parent / "outcome_tree.txt"
    write_tree_txt(data, solver, leaves, out)
    print(f"Листьев: {summary['n_leaves']}")
    print(f"E[W]={summary['E_W']:.4f}, E[I]={summary['E_income']:.4f}")
    print(f"E[r]={summary['E_return']*100:.4f}%, P(loss)={summary['P_loss']:.4f}")
    print(f"minI={summary['min_income']:.4f}, maxI={summary['max_income']:.4f}")
    print(f"Файл: {out.name}")


if __name__ == "__main__":
    main()
