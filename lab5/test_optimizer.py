import ast
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tokenize
import unittest
from pathlib import Path

import numpy as np

from functions import get_function
from optimizer import Budget, anneal, optimize, reflect, validate_params
from run import run_once


ROOT = Path(__file__).resolve().parent


class OptimizerTests(unittest.TestCase):
    def test_external_budget_and_bounds(self):
        for mode in ("hybrid", "sa", "tabu"):
            for limit in (1, 2, 7, 63, 64, 65, 100, 257, 1000, 10000):
                with self.subTest(mode=mode, limit=limit):
                    calls = []

                    def objective(x):
                        self.assertLess(len(calls), limit)
                        self.assertEqual(x.shape, (2,))
                        self.assertTrue(np.all(x >= [-4.8, -5.0]))
                        self.assertTrue(np.all(x <= [5.3, 5.1]))
                        calls.append(x.copy())
                        return float((x[0] - 0.371) ** 2 + (x[1] + 0.619) ** 2 - 20)

                    result = optimize(objective, [(-4.8, 5.3), (-5.0, 5.1)], limit, 27, mode=mode)
                    self.assertEqual(result["nfe"], len(calls))
                    self.assertEqual(len(result["history"]), len(calls))
                    self.assertTrue(np.all(np.diff(result["history"]) <= 0))
                    expected = (result["x"][0] - 0.371) ** 2 + (result["x"][1] + 0.619) ** 2 - 20
                    self.assertEqual(result["f"], expected)
                    self.assertEqual(result["history"][-1], result["f"])
                    self.assertTrue(any(np.array_equal(result["x"], x) for x in calls))

    def test_repeatability_and_global_rng(self):
        state = np.random.get_state()
        tf = get_function("levy13")
        first = optimize(tf, tf.bounds, seed=27)
        second = optimize(tf, tf.bounds, seed=27)
        np.testing.assert_array_equal(first["x"], second["x"])
        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["switches"], second["switches"])
        self.assertEqual(first["stats"], second["stats"])
        current = np.random.get_state()
        self.assertEqual(state[0], current[0])
        np.testing.assert_array_equal(state[1], current[1])
        self.assertEqual(state[2:], current[2:])

    def test_independent_levy_seeds(self):
        for seed in range(101, 121):
            with self.subTest(seed=seed):
                tf = get_function(f"levy13~{seed}")
                row, _ = run_once(tf, seed, 10000)
                self.assertLessEqual(row["e"], 1e-8)

    def test_private_function(self):
        angle = 0.713
        rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        optimum = np.array([2.317, -1.119])

        def private(x):
            z = rotation @ (x - optimum)
            return float(z @ z + 8 * np.sum(np.sin(3 * z) ** 2) - 17)

        for seed in (41, 42, 43):
            result = optimize(private, [(-4, 5), (-3, 2)], seed=seed)
            self.assertAlmostEqual(result["f"], -17, places=7)
            np.testing.assert_allclose(result["x"], optimum, atol=1e-4)

    def test_boundary_minimum(self):
        result = optimize(lambda x: float(x[0] - 2 * x[1]), [(-2, 3), (-4, 7)], seed=7)
        np.testing.assert_allclose(result["x"], [-2, 7], atol=1e-7)
        self.assertAlmostEqual(result["f"], -16, places=7)

    def test_flat_and_small_domain(self):
        result = optimize(lambda x: 4.0, [(1, 1 + 1e-10), (-2, -2 + 1e-10)], 129, 1)
        self.assertEqual(result["f"], 4.0)
        self.assertLessEqual(result["nfe"], 129)

    def test_both_components_do_work(self):
        tf = get_function("levy13")
        result = optimize(tf, tf.bounds, seed=1)
        self.assertGreater(result["stats"]["sa_uphill"], 0)
        self.assertGreater(result["stats"]["tabu_uphill"], 0)
        self.assertGreater(result["stats"]["tabu_rejected"], 0)
        self.assertGreater(result["stats"]["basins"], 0)
        stages = result["switches"]
        lengths = [(stages[i][1], stages[i + 1][0] - stages[i][0]) for i in range(len(stages) - 1)]
        self.assertTrue(any(name.startswith("Отжиг") and n > 0 for name, n in lengths))
        self.assertTrue(any(name.startswith("Запреты") and n > 0 for name, n in lengths))

    def test_component_modes(self):
        tf = get_function("levy13")
        sa = optimize(tf, tf.bounds, max_nfe=2000, seed=2, mode="sa")
        tabu = optimize(tf, tf.bounds, max_nfe=2000, seed=2, mode="tabu")
        self.assertEqual(sa["stats"]["tabu_uphill"], 0)
        self.assertEqual(sa["stats"]["tabu_rejected"], 0)
        self.assertEqual(tabu["stats"]["sa_uphill"], 0)

    def test_tabu_aspiration(self):
        p = validate_params({"tabu_radius": 0.99})
        budget = Budget(lambda x: float(x @ x), [(0, 1), (0, 1)], 101)
        start = np.array([0.5, 0.5])
        value = budget.eval(start)
        anneal(budget, np.random.default_rng(9), start, value, 100, 1.0, [start], p)
        self.assertGreater(budget.stats["tabu_rejected"], 0)
        self.assertGreater(budget.stats["aspirations"], 0)
        self.assertLess(budget.best_f, value)

    def test_invalid_inputs(self):
        good = [(-1, 1), (-1, 1)]
        for bounds in ([(0, 0), (-1, 1)], [(2, 1), (-1, 1)], [(0, float("inf")), (0, 1)], [(0, 1)]):
            with self.assertRaises(ValueError):
                optimize(lambda x: 0, bounds)
        for limit in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                optimize(lambda x: 0, good, limit)
        for params in ({"cycles": 0}, {"mode": "wrong"}, {"cooling_ratio": 1},
                       {"tabu_radius": -1}, {"minimum_step": 0.5}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                optimize(lambda x: 0, good, **params)
        with self.assertRaises(ValueError):
            optimize(lambda x: 0, good, constraints=[lambda x: x[0]])

    def test_invalid_values(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                optimize(lambda x: value, [(-1, 1), (-1, 1)])

    def test_reflection(self):
        np.testing.assert_allclose(reflect(np.array([-0.2, 1.2])), [0.2, 0.8])
        points = reflect(np.array([-100.2, 100.2]))
        self.assertTrue(np.all((points >= 0) & (points <= 1)))

    def test_no_output_or_mutation(self):
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        original = bounds.copy()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            optimize(lambda x: float(x @ x), bounds, 50, 1)
        self.assertEqual(output.getvalue(), "")
        np.testing.assert_array_equal(bounds, original)

    def test_runner_recounts_and_cli(self):
        process = subprocess.run([sys.executable, str(ROOT / "run.py"), "--nfe", "100", "--seed", "27"],
                                 cwd=ROOT, text=True, capture_output=True, timeout=60, check=True)
        line = next(line for line in process.stdout.splitlines() if line.startswith("JSON:"))
        row = json.loads(line.removeprefix("JSON:"))
        self.assertLessEqual(row["nfe"], 100)
        self.assertEqual(row["seed"], 27)
        self.assertEqual(row["func"], "levy13")
        self.assertGreater(row["time_s"], 0)

    def test_history_and_snapshots(self):
        row, result = run_once(get_function("levy13"), 1, 10000)
        self.assertEqual(result["trajectory"][-1][0], next(i + 1 for i, f in enumerate(result["history"]) if f == row["f"]))
        self.assertGreaterEqual(len(result["snapshots"]), 4)
        for nfe, _, points in result["snapshots"]:
            self.assertLessEqual(nfe, row["nfe"])
            self.assertEqual(points.shape[1], 2)
            self.assertTrue(np.all((points >= -10) & (points <= 10)))

    def test_own_code_style_and_imports(self):
        for path in ROOT.glob("*.py"):
            if path.name == "functions.py":
                continue
            source = path.read_text()
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens), path.name)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertIsNone(ast.get_docstring(node), path.name)
        tree = ast.parse((ROOT / "optimizer.py").read_text())
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertLessEqual(modules, {"math", "collections", "numpy"})

    def test_reference_functions_unchanged(self):
        digest = hashlib.sha256((ROOT / "functions.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "ba3ef9c13b81a4f36f8e1ac37b662e1c8beff9c0813b4be301e0c5aea90506ce")


if __name__ == "__main__":
    unittest.main()
