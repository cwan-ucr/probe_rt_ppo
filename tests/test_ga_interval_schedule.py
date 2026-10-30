"""Fast checks for the 31 x 10 open-loop GA schedule."""

import unittest

import numpy as np

from src.rl import train_ga_static_offset as ga

C = ga.C


class FakeEnv:
    episode_end = 28800

    def __init__(self):
        self.baseline = None
        self.actions = []

    def reset(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return None, 0.0, len(self.actions) == 31, {"total_delay_vehh": 1.0}


class GAIntervalScheduleTests(unittest.TestCase):
    def test_interval_genome_applies_31_distinct_action_rows(self):
        env = FakeEnv()
        matrix = np.zeros((31, C.N_AGENTS), dtype=int)
        matrix[0, 0] = -5
        matrix[30, 9] = 7
        delay, _info = ga._rollout(env, tuple(matrix.reshape(-1)))
        self.assertEqual(delay, 31.0)
        self.assertEqual(len(env.actions), 31)
        self.assertAlmostEqual(env.actions[0][0], -5 / C.ACTION_HALF_RANGE)
        self.assertAlmostEqual(env.actions[30][9], 7 / C.ACTION_HALF_RANGE)
        self.assertEqual(env.actions[1][0], 0.0)

    def test_static_genome_repeats_one_action_row(self):
        env = FakeEnv()
        ga._rollout(env, (3,) + (0,) * (C.N_AGENTS - 1))
        self.assertEqual(len(env.actions), 31)
        self.assertTrue(all(action[0] == env.actions[0][0]
                            for action in env.actions))

    def test_interval_population_and_offspring_have_310_genes(self):
        rng = np.random.default_rng(42)
        n_genes = 31 * C.N_AGENTS
        self.assertEqual(n_genes, 310)
        population = ga._initial_population(24, rng, n_genes, "interval")
        self.assertEqual(len(population), 24)
        self.assertEqual(len(set(population)), 24)
        self.assertTrue(all(len(genome) == n_genes for genome in population))
        scores = np.arange(24, dtype=float)
        offspring = ga._next_population(
            population, scores, rng, elite_count=2,
            mutation_rate=0.01, mutation_sigma=5.0, n_genes=n_genes)
        self.assertEqual(len(offspring), 24)
        self.assertEqual(offspring[0], (0,) * n_genes)
        self.assertTrue(all(len(genome) == n_genes for genome in offspring))


if __name__ == "__main__":
    unittest.main()
