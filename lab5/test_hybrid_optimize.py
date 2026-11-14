import ast
import io
import math
import tokenize
import unittest
from pathlib import Path

from hybrid_optimize import DEFAULT_BOUNDS, EvalCounter, hybrid_optimize, nelder_mead_2d, rastrigin2d


class HybridTests(unittest.TestCase):
    def test_even_grid_does_not_contain_zero(self):
        for low, high in ((-5.12, 5.12), (-4.8, 5.3), (-5.0, 5.1)):
            values = [low + (high - low) * index / 9 for index in range(10)]
            self.assertFalse(any(abs(value) < 1e-12 for value in values))

    def test_rastrigin_across_seeds_and_shifted_boxes(self):
        for bounds in (DEFAULT_BOUNDS, (-4.8, 5.3, -5.0, 5.1)):
            for seed in range(20):
                with self.subTest(bounds=bounds, seed=seed):
                    result = hybrid_optimize(rastrigin2d, bounds, seed=seed, lower_bound=0.0)
                    self.assertLess(result.f, 1e-12)
                    self.assertLess(math.hypot(result.x, result.y), 1e-7)
                    self.assertEqual(result.f, rastrigin2d(result.x, result.y))
                    self.assertEqual(result.status, 'target_reached')

    def test_translated_and_rotated_objectives(self):
        for angle in (0.0, 0.21, 0.45, 0.6435011088, 1.1):
            cosine, sine = math.cos(angle), math.sin(angle)

            def objective(x, y):
                x, y = x - 0.37, y + 0.23
                return rastrigin2d(cosine * x - sine * y, sine * x + cosine * y)

            for seed in range(20):
                with self.subTest(angle=angle, seed=seed):
                    result = hybrid_optimize(objective, (-4.8, 5.3, -5.0, 5.1), seed=seed, lower_bound=0.0)
                    self.assertLess(result.f, 1e-12)
                    self.assertLess(math.hypot(result.x - 0.37, result.y + 0.23), 1e-7)

    def test_different_grid_sizes(self):
        for grid in (6, 8, 10, 12, 14):
            for bounds in (DEFAULT_BOUNDS, (-4.8, 5.3, -5.0, 5.1)):
                result = hybrid_optimize(rastrigin2d, bounds, grid=grid, lower_bound=0.0)
                self.assertLess(result.f, 1e-12)

    def test_no_known_lower_bound(self):
        result = hybrid_optimize(rastrigin2d, (-4.8, 5.3, -5.0, 5.1))
        self.assertLess(result.f, 1e-12)
        self.assertEqual(result.status, 'search_complete')

    def test_negative_values_do_not_trigger_early_success(self):
        def objective(x, y):
            return (x - 1.37) ** 2 + 3 * (y + 0.61) ** 2 - 100

        result = hybrid_optimize(objective, (-4, 4, -4, 4))
        self.assertLess(abs(result.f + 100), 1e-10)
        self.assertLess(math.hypot(result.x - 1.37, result.y + 0.61), 1e-6)
        self.assertEqual(result.status, 'search_complete')

    def test_rosenbrock(self):
        result = hybrid_optimize(lambda x, y: (1 - x) ** 2 + 100 * (y - x * x) ** 2, (-2, 2, -1, 3), lower_bound=0)
        self.assertLess(result.f, 1e-12)

    def test_boundary_initial_simplex_is_not_collapsed(self):
        fcount = EvalCounter(lambda x, y: (x - 0.37) ** 2 + (y - 0.23) ** 2)
        x, y, value = nelder_mead_2d(fcount, 1, 1, (0, 1, 0, 1))
        self.assertLess(value, 1e-14)
        self.assertAlmostEqual(x, 0.37, places=7)
        self.assertAlmostEqual(y, 0.23, places=7)

    def test_boundary_optimum_and_all_evaluations_stay_in_box(self):
        evaluations = []

        def objective(x, y):
            self.assertTrue(1 <= x <= 2 and -3 <= y <= -1)
            evaluations.append((x, y))
            return (x - 4) ** 2 + (y + 2.37) ** 2

        result = hybrid_optimize(objective, (1, 2, -3, -1))
        self.assertAlmostEqual(result.x, 2, places=7)
        self.assertAlmostEqual(result.y, -2.37, places=6)
        self.assertAlmostEqual(result.f, 4, places=12)
        self.assertEqual(result.n_eval, len(evaluations))
        self.assertEqual(result.n_eval, len(set(evaluations)))

    def test_constant_function_and_small_box(self):
        result = hybrid_optimize(lambda x, y: 7.0, (0.01, 0.02, -0.03, -0.01))
        self.assertEqual(result.f, 7.0)
        self.assertEqual(result.status, 'search_complete')

    def test_exact_cache_counts_real_calls(self):
        calls = []
        counter = EvalCounter(lambda x, y: calls.append((x, y)) or x * x + y * y)
        self.assertEqual(counter(1, 2), 5)
        self.assertEqual(counter(1, 2), 5)
        self.assertEqual(counter.n, 1)
        counter(1 + 1e-12, 2)
        self.assertEqual(counter.n, 2)
        self.assertEqual(len(calls), 2)

    def test_reproducibility(self):
        first = hybrid_optimize(rastrigin2d, lower_bound=0)
        second = hybrid_optimize(rastrigin2d, lower_bound=0)
        self.assertEqual((first.x, first.y, first.f, first.n_eval), (second.x, second.y, second.f, second.n_eval))

    def test_independent_randomized_stress_suite(self):
        from benchmark import stress_check

        result = stress_check()
        self.assertEqual(result['successes'], 100)

    def test_invalid_parameters_and_function_values(self):
        for bounds in ((1, 0, 0, 1), (0, 1, 2, 2), (0, math.inf, 0, 1)):
            with self.assertRaises(ValueError):
                hybrid_optimize(rastrigin2d, bounds)
        for options in ({'grid': 1}, {'top_k': 0}, {'random_points': -1}, {'line_budget': 0}, {'angles': 0}, {'local_step': 0}, {'target_tol': math.nan}, {'strategy': 'unknown'}, {'lower_bound': math.inf}):
            with self.assertRaises(ValueError):
                hybrid_optimize(rastrigin2d, **options)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                hybrid_optimize(lambda x, y: value)
        with self.assertRaises(ValueError):
            hybrid_optimize(lambda x, y: -1, lower_bound=0)

    def test_solver_has_no_comments_docstrings_or_external_solvers(self):
        source = Path(__file__).with_name('hybrid_optimize.py').read_text()
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.assertIsNone(ast.get_docstring(node))
            if isinstance(node, ast.Import):
                self.assertFalse(any('scipy' in alias.name for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn('scipy', node.module or '')


if __name__ == '__main__':
    unittest.main(verbosity=2)
