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
    e_w = sum(L.probability * L.final_wealth for L in leaves)
    e_i = sum(L.probability * L.income for L in leaves)
    e_r = sum(L.probability * L.return_rate for L in leaves)
    p_loss = sum(L.probability for L in leaves if L.income < 0)
    return {
        "n_leaves": len(leaves),
        "E_W": e_w,
        "E_income": e_i,
        "E_return": e_r,
        "P_loss": p_loss,
        "min_income": min(L.income for L in leaves),
        "max_income": max(L.income for L in leaves),
    }


def write_reports(
    data: ProblemData, solver: BellmanSolver, leaves: list[Leaf], out_dir: Path
) -> None:
    summary = summarize(leaves)
    lines = []
    lines.append("# Дерево исходов оптимальной стратегии")
    lines.append("")
    lines.append(
        f"W0 = {rub(data.w0):g} д.е., число листьев = {summary['n_leaves']} "
        f"(= 3^{len(data.stages)})."
    )
    lines.append("")
    lines.append("## Сводка")
    lines.append("")
    lines.append(f"- E[W_final] = {summary['E_W']:.4f}")
    lines.append(f"- E[чистый доход] = {summary['E_income']:.4f}")
    lines.append(f"- E[доходность] = {summary['E_return']*100:.4f}%")
    lines.append(f"- P(убыток) = {summary['P_loss']:.4f}")
    lines.append(f"- min доход = {summary['min_income']:.4f}")
    lines.append(f"- max доход = {summary['max_income']:.4f}")
    lines.append("")
    lines.append("## Листья")
    lines.append("")
    for i, L in enumerate(leaves, start=1):
        seq = " → ".join(SCENARIO_NAMES[s] for s in L.scenarios)
        ops = " | ".join(format_action(a, data) for a in L.actions)
        fees = ", ".join(f"{c:.2f}" for c in L.commissions)
        lines.append(f"### Лист {i}")
        lines.append(f"- Сценарии: {seq}")
        lines.append(f"- Вероятность: {L.probability:.6f}")
        lines.append(f"- Операции: {ops}")
        lines.append(f"- Комиссии по этапам: {fees} (сумма {sum(L.commissions):.2f})")
        lines.append(f"- Конечный капитал W = {L.final_wealth:.4f}")
        lines.append(f"- Чистый доход = {L.income:.4f}")
        lines.append(f"- Доходность = {L.return_rate*100:.4f}%")
        lines.append("")

    (out_dir / "outcome_tree.md").write_text("\n".join(lines), encoding="utf-8")

    csv_lines = [
        "leaf,scenarios,probability,actions,commission_sum,W_final,income,return"
    ]
    for i, L in enumerate(leaves, start=1):
        seq = "-".join(str(s) for s in L.scenarios)
        acts = ";".join(str(a.as_tuple()) for a in L.actions)
        csv_lines.append(
            f"{i},{seq},{L.probability:.8f},\"{acts}\",{sum(L.commissions):.4f},"
            f"{L.final_wealth:.4f},{L.income:.4f},{L.return_rate:.8f}"
        )
    (out_dir / "outcome_tree.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    text = []
    text.append("ДЕРЕВО ИСХОДОВ (оптимальная политика, критерий ожидания)")
    text.append(f"s0 = {format_state(data.initial)}")
    text.append("")

    def print_node(stage_idx: int, state: State, prefix: str, prob: float) -> None:
        if stage_idx >= len(data.stages):
            income = rub(state.wealth()) - rub(data.w0)
            text.append(
                f"{prefix}ЛИСТ W={rub(state.wealth()):g}, доход={income:.2f}, P={prob:.6f}"
            )
            return
        key = (stage_idx, state)
        action = solver.policy_cache[key]
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
    text.append(
        f"E[W]={summary['E_W']:.4f}, E[I]={summary['E_income']:.4f}, "
        f"E[r]={summary['E_return']*100:.4f}%, P(loss)={summary['P_loss']:.4f}, "
        f"minI={summary['min_income']:.4f}, maxI={summary['max_income']:.4f}"
    )
    (out_dir / "outcome_tree.txt").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    data = default_problem()
    solver = BellmanSolver(data, "expectation")
    print("Считаю оптимальную политику...")
    res = solver.solve()
    print(f"I* = {res.best_income:.4f}")
    leaves = build_outcome_tree(data, solver)
    summary = summarize(leaves)
    out = Path(__file__).resolve().parent
    write_reports(data, solver, leaves, out)
    print(f"Листьев: {summary['n_leaves']}")
    print(f"E[W]={summary['E_W']:.4f}, E[I]={summary['E_income']:.4f}")
    print(f"E[r]={summary['E_return']*100:.4f}%, P(loss)={summary['P_loss']:.4f}")
    print(f"minI={summary['min_income']:.4f}, maxI={summary['max_income']:.4f}")
    print("Файлы: outcome_tree.md, outcome_tree.csv, outcome_tree.txt")


if __name__ == "__main__":
    main()
