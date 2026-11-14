import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

from hybrid_optimize import DEFAULT_BOUNDS, hybrid_optimize, rastrigin2d


BASELINE_COMMIT = '44fb275'


def load_baseline():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ['git', 'show', f'{BASELINE_COMMIT}:lab5/hybrid_optimize.py'], cwd=root, text=True,
    )
    module = types.ModuleType('lab5_baseline')
    module.__file__ = f'{BASELINE_COMMIT}:lab5/hybrid_optimize.py'
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module.hybrid_optimize


def translated_rastrigin(x, y):
    return rastrigin2d(x - 0.37, y + 0.23)


def rotated_rastrigin(x, y):
    x, y = x - 0.37, y + 0.23
    return rastrigin2d(0.8 * x - 0.6 * y, 0.6 * x + 0.8 * y)


def stress_check():
    rng = random.Random(20261115)
    records = []
    for seed in range(100):
        shift_x, shift_y, angle = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, math.pi)
        cosine, sine = math.cos(angle), math.sin(angle)

        def objective(x, y):
            x, y = x - shift_x, y - shift_y
            return rastrigin2d(cosine * x - sine * y, sine * x + cosine * y)

        result = hybrid_optimize(objective, (-4.8, 5.3, -5.0, 5.1), seed=seed, lower_bound=0.0)
        records.append({
            'seed': seed, 'shift_x': shift_x, 'shift_y': shift_y, 'angle': angle,
            'f': result.f, 'seconds': result.elapsed_sec, 'n_eval': result.n_eval,
        })
    return {
        'generator_seed': 20261115,
        'successes': sum(record['f'] <= 1e-12 for record in records),
        'trials': len(records),
        'f_max': max(record['f'] for record in records),
        'seconds_median': statistics.median(record['seconds'] for record in records),
        'seconds_max': max(record['seconds'] for record in records),
        'runs': records,
    }


def run_benchmark(repeats, include_baseline):
    cases = {
        'standard': (rastrigin2d, DEFAULT_BOUNDS),
        'shifted_box': (rastrigin2d, (-4.8, 5.3, -5.0, 5.1)),
        'translated': (translated_rastrigin, (-4.8, 5.3, -5.0, 5.1)),
        'rotated': (rotated_rastrigin, (-4.8, 5.3, -5.0, 5.1)),
    }
    solvers = {
        name: (lambda fun, bounds, selected=name: hybrid_optimize(fun, bounds, strategy=selected, lower_bound=0.0))
        for name in ('multistart', 'axes', 'guarded')
    }
    if include_baseline:
        solvers['baseline'] = load_baseline()
    output = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'repeats': repeats,
        'seed': 0,
        'grid': 10,
        'lower_bound': 0.0,
        'baseline_commit': BASELINE_COMMIT if include_baseline else None,
        'timing': 'perf_counter around solver, includes objective evaluations, excludes imports and visualization',
        'cases': {},
    }
    order = random.Random(42)
    for name, (fun, bounds) in cases.items():
        records = {variant: [] for variant in solvers}
        for solver in solvers.values():
            solver(fun, bounds)
        for _ in range(repeats):
            variants = list(solvers)
            order.shuffle(variants)
            for variant in variants:
                start = time.perf_counter()
                result = solvers[variant](fun, bounds)
                elapsed = time.perf_counter() - start
                if not bounds[0] <= result.x <= bounds[1] or not bounds[2] <= result.y <= bounds[3]:
                    raise AssertionError('Результат вышел за границы области.')
                if not math.isclose(result.f, fun(result.x, result.y), abs_tol=1e-14):
                    raise AssertionError('Значение функции не соответствует найденной точке.')
                records[variant].append({'x': result.x, 'y': result.y, 'f': result.f, 'seconds': elapsed, 'n_eval': result.n_eval})
        output['cases'][name] = {'bounds': bounds, 'variants': {}}
        for variant, runs in records.items():
            summary = {
                'successes': sum(abs(run['f']) <= 1e-12 for run in runs),
                'f_min': min(run['f'] for run in runs),
                'f_max': max(run['f'] for run in runs),
                'seconds_min': min(run['seconds'] for run in runs),
                'seconds_median': statistics.median(run['seconds'] for run in runs),
                'seconds_max': max(run['seconds'] for run in runs),
                'n_eval_median': statistics.median(run['n_eval'] for run in runs),
                'runs': runs,
            }
            output['cases'][name]['variants'][variant] = summary
            print(f'{name:12} {variant:12} successes={summary["successes"]}/{repeats} f_max={summary["f_max"]:.3e} median={summary["seconds_median"]:.6f} s calls={summary["n_eval_median"]}', flush=True)
    output['stress'] = stress_check()
    print(f'guarded stress: {output["stress"]["successes"]} successful trials out of {output["stress"]["trials"]}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description='Воспроизводимое сравнение вариантов гибрида')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('benchmark_results.json'))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Число повторов должно быть положительным.')
    result = run_benchmark(args.repeats, args.baseline)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Результаты сохранены: {args.output.name}')


if __name__ == '__main__':
    main()
