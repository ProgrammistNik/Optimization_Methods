import argparse
import json
import time

import numpy as np

from functions import get_function
from optimizer import optimize


class BudgetExceeded(BaseException):
    pass


class CountedF:
    def __init__(self, tf, max_nfe):
        self.tf = tf
        self.max_nfe = max_nfe
        self.nfe = 0
        self.lo = tf.lower
        self.hi = tf.upper

    def __call__(self, x):
        self.nfe += 1
        if self.nfe > self.max_nfe:
            raise BudgetExceeded("Превышен бюджет вычислений")
        x = np.asarray(x, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)) or np.any(x < self.lo) or np.any(x > self.hi):
            raise ValueError("Недопустимая точка при вычислении функции")
        return self.tf(x)


def run_once(tf, seed, nfe, **params):
    counted = CountedF(tf, nfe)
    options = {"constraints": list(tf.constraints)} if tf.constraints else {}
    started = time.perf_counter()
    result = optimize(counted, list(tf.bounds), max_nfe=nfe, seed=seed, **options, **params)
    elapsed = time.perf_counter() - started
    x = np.asarray(result["x"], dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)) or not tf.is_feasible(x, tol=0):
        raise ValueError("Оптимизатор вернул недопустимую точку")
    fx = tf(x)
    if result["nfe"] != counted.nfe or not np.isclose(fx, result["f"], rtol=1e-12, atol=1e-14):
        raise ValueError("Независимая проверка результата не пройдена")
    error = tf.error(fx)
    row = dict(func=tf.name, seed=seed, x1=float(x[0]), x2=float(x[1]), f=fx,
               e=error, success=bool(error <= tf.eps), nfe=counted.nfe, time_s=elapsed)
    return row, result


def summarize(rows):
    values = np.array([r["f"] for r in rows])
    errors = np.array([max(r["e"], 1e-12) for r in rows])
    logs = np.log10(errors)
    times = np.array([r["time_s"] for r in rows])
    return dict(best=float(values.min()), median=float(np.median(values)), worst=float(values.max()),
                success_rate=float(np.mean([r["success"] for r in rows])),
                median_log_error=float(np.median(logs)),
                iqr_log_error=float(np.percentile(logs, 75) - np.percentile(logs, 25)),
                mean_time_s=float(times.mean()), median_time_s=float(np.median(times)),
                mean_nfe=float(np.mean([r["nfe"] for r in rows])))


def print_row(row):
    state = "успех" if row["success"] else "порог точности не достигнут"
    print(f"seed={row['seed']}  x* = ({row['x1']:.12g}, {row['x2']:.12g})  "
          f"f(x*) = {row['f']:.12g}  NFE = {row['nfe']}  время = {row['time_s']:.6f} с  {state}")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("нужно положительное целое число")
    return number


def main():
    parser = argparse.ArgumentParser(description="Битва методов оптимизации, вариант 7")
    parser.add_argument("--func", default="levy13")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nfe", type=positive_int, default=10000)
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument("--mode", choices=("hybrid", "sa", "tabu"), default="hybrid")
    args = parser.parse_args()
    tf = get_function(args.func)
    run_once(tf, args.seed, args.nfe, mode=args.mode)
    rows = []
    for seed in range(args.seed, args.seed + args.runs):
        row, _ = run_once(tf, seed, args.nfe, mode=args.mode)
        rows.append(row)
        print_row(row)
    if args.runs == 1:
        print("JSON:", json.dumps(rows[0], ensure_ascii=False))
    else:
        summary = summarize(rows)
        print("\n| Лучшее f | Медиана f | Худшее f | Доля успехов | Среднее время, с | Среднее NFE |")
        print("| --- | --- | --- | --- | --- | --- |")
        print(f"| {summary['best']:.6g} | {summary['median']:.6g} | {summary['worst']:.6g} | "
              f"{summary['success_rate']:.2%} | {summary['mean_time_s']:.6f} | {summary['mean_nfe']:.1f} |")
        print("JSON:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
