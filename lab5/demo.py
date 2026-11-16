import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmark import compare, fingerprint, metadata, save
from functions import get_function
from run import positive_int, print_row, run_once


ROOT = Path(__file__).resolve().parent
COLORS = ("#16796f", "#d66b30", "#526cc0")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 14, "axes.titleweight": "bold",
                         "axes.labelcolor": "#334155", "text.color": "#172b37",
                         "axes.edgecolor": "#cbd5e1", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "#f8fafb", "savefig.facecolor": "white"})


def grid(tf, n=240):
    x, y = np.meshgrid(np.linspace(*tf.bounds[0], n), np.linspace(*tf.bounds[1], n))
    z = np.array([tf(point) for point in zip(x.ravel(), y.ravel())]).reshape(x.shape)
    return x, y, z


def finish(fig, destination):
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def contour(tf, result, x, y, z, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    snapshots = result["snapshots"]
    indices = np.linspace(0, len(snapshots) - 1, 4).round().astype(int)
    levels = np.linspace(np.log10(max(1e-9, (z - tf.f_star).min())),
                         np.log10(max(1e-8, (z - tf.f_star).max())), 36)
    for ax, index in zip(axes.ravel(), indices):
        nfe, name, points = snapshots[index]
        image = ax.contourf(x, y, np.log10(np.maximum(z - tf.f_star, 1e-9)), levels=levels,
                            cmap="Blues", alpha=0.9, extend="both")
        trajectory = np.array([point for count, point in result["trajectory"] if count <= nfe])
        if len(trajectory):
            ax.plot(*trajectory.T, color=COLORS[0], lw=1.6, alpha=0.85)
            ax.scatter(*trajectory[-1], marker="x", color=COLORS[0], s=90, linewidth=2.2,
                       label="Лучшее решение")
        ax.scatter(*points.T, s=32, color=COLORS[1], edgecolor="white", linewidth=0.5,
                   label="Состояние поиска", zorder=3)
        for point in tf.x_star:
            ax.scatter(*point, marker="*", s=160, color="#172b37", edgecolor="white",
                       linewidth=0.7, zorder=4)
        ax.set(title=f"{name} · {nfe} вычислений", xlabel="x₁", ylabel="x₂",
               xlim=tf.bounds[0], ylim=tf.bounds[1], aspect="equal")
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.colorbar(image, ax=axes, shrink=0.75, label="lg ошибки на сетке")
    fig.suptitle("Леви N.13. Ход поиска" if tf.name == "levy13" else tf.title, fontsize=20, weight="bold")
    finish(fig, output / "contour.png")


def convergence(tf, result, output):
    errors = np.maximum((np.asarray(result["history"]) - tf.f_star) / max(1, abs(tf.f_star)), 1e-20)
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    stages = result["switches"]
    used = set()
    for i, (start, name) in enumerate(stages):
        end = stages[i + 1][0] if i + 1 < len(stages) else result["nfe"]
        category = "Отжиг" if name.startswith("Отжиг") else "Поиск с запретами"
        if name == "Начальная выборка":
            continue
        color = COLORS[1] if category == "Отжиг" else COLORS[0]
        ax.axvspan(start, end, color=color, alpha=0.08, label=category if category not in used else None)
        ax.axvline(start, color=color, lw=0.6, alpha=0.5, linestyle="--")
        used.add(category)
    ax.semilogy(np.arange(1, len(errors) + 1), errors, color=COLORS[0], lw=1.8, label="Лучшее значение")
    ax.axhline(tf.eps, color=COLORS[2], linestyle=":", lw=1.5, label="Порог успеха")
    ax.set(title="Точность по мере расходования бюджета", xlabel="Число вычислений функции",
           ylabel="Нормированная ошибка", xlim=(0, result["nfe"]))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    finish(fig, output / "convergence.png")


def surface(tf, x, y, z, output):
    fig = plt.figure(figsize=(10, 6.5), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(x, y, z, cmap="Blues", rcount=100, ccount=100, linewidth=0, antialiased=True)
    ax.view_init(elev=35, azim=-55)
    ax.set(title=tf.title, xlabel="x₁", ylabel="x₂", zlabel="f(x)")
    fig.savefig(output / "surface.png", dpi=170, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


def boxplot(comparison, runs, nfe, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    modes = ("hybrid", "sa", "tabu")
    data = [[np.log10(max(row["e"], 1e-20)) for row in comparison[mode]["rows"]] for mode in modes]
    artists = axes[0].boxplot(data, patch_artist=True, widths=0.5,
                              medianprops={"color": "#172b37", "linewidth": 1.5})
    for patch, color in zip(artists["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for index, values in enumerate(data, 1):
        jitter = np.linspace(-0.13, 0.13, len(values))
        axes[0].scatter(index + jitter, values, color=COLORS[index - 1], s=13, alpha=0.6)
    axes[0].set(title="Распределение ошибки", ylabel="lg нормированной ошибки")
    times = [comparison[mode]["summary"]["median_time_s"] * 1000 for mode in modes]
    bars = axes[1].bar(range(1, 4), times, color=COLORS, width=0.55, alpha=0.85)
    axes[1].bar_label(bars, fmt="%.1f", padding=4)
    axes[1].set(title="Медианное время", ylabel="Миллисекунды", ylim=(0, max(times) * 1.25))
    for ax in axes:
        ax.set_xticks([1, 2, 3], ["Гибрид", "Отжиг", "Запреты"])
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"Одинаковые {runs} зёрен и лимит {nfe} вычислений", fontsize=17, weight="bold")
    finish(fig, output / "boxplot.png")


def step_sensitivity(entries, output):
    steps = [float(step) for step in entries]
    rates = [entry["summary"]["success_rate"] * 100 for entry in entries.values()]
    errors = [max(entry["summary"]["median"], 1e-20) for entry in entries.values()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].semilogx(steps, rates, "o-", color=COLORS[0])
    axes[0].set(ylabel="Успешные запуски, %", ylim=(0, 105), title="Влияние начального шага")
    axes[1].loglog(steps, errors, "o-", color=COLORS[1])
    axes[1].set(ylabel="Медианная ошибка", title="Точность при разном шаге")
    for ax in axes:
        ax.set_xlabel("Начальный шаг в нормированных координатах")
        ax.grid(alpha=0.18)
    finish(fig, output / "step_sensitivity.png")


def main():
    parser = argparse.ArgumentParser(description="Рисунки для отчёта по варианту 7")
    parser.add_argument("--func", default="levy13")
    parser.add_argument("--runs", type=positive_int, default=30)
    parser.add_argument("--nfe", type=positive_int, default=10000)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "img")
    args = parser.parse_args()
    tf = get_function(args.func)
    if args.results:
        saved = json.loads(args.results.read_text())
        meta = saved["metadata"]
        if meta["optimizer_sha256"] != fingerprint() or meta["runs"] != args.runs or meta["max_nfe"] != args.nfe:
            raise ValueError("Результаты не соответствуют текущему коду или параметрам серии")
        comparison = saved["comparison"][args.func]
    else:
        comparison = compare(tf, args.runs, args.nfe)
        saved = dict(metadata=metadata(args.runs, args.nfe), comparison={args.func: comparison})
        save(ROOT / "results" / "demo.json", saved)
    run_once(tf, 1, args.nfe)
    row, result = run_once(tf, 1, args.nfe)
    print_row(row)
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    x, y, z = grid(tf)
    contour(tf, result, x, y, z, args.output)
    convergence(tf, result, args.output)
    surface(tf, x, y, z, args.output)
    boxplot(comparison, args.runs, args.nfe, args.output)
    if saved.get("step_sensitivity"):
        step_sensitivity(saved["step_sensitivity"], args.output)
    print("Рисунки сохранены:", args.output)


if __name__ == "__main__":
    main()
