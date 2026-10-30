"""
Genetic Algorithm optimizer for demand calibration.

Better than SPSA for this problem because:
  - Searches globally (multiple candidates per generation, no gradient assumption)
  - Tolerates noisy simulation objective
  - Handles bounded flow parameters naturally via clamping in mutation/crossover
"""

import os
import json
import numpy as np
from datetime import datetime


class GAOptimizer:
    """Genetic algorithm for FlowParameters / DemandParameters calibration."""

    def __init__(self, objective_fn, initial_params,
                 population_size=12, n_generations=20,
                 elite_frac=0.25, mutation_rate=0.2, mutation_sigma=0.15,
                 crossover_rate=0.7, init_jitter=0.25,
                 output_dir=None, phase_name="", seed=42,
                 batch_objective_fn=None):
        """
        Parameters
        ----------
        objective_fn : callable
            Maps params → float (lower is better).
        initial_params : FlowParameters or DemandParameters
            Seed individual; defines vector length, bounds, and reconstructor.
        population_size : int
            Individuals per generation. Each costs 1 simulation.
        n_generations : int
            Total generations.
        elite_frac : float
            Fraction of best individuals carried over unchanged.
        mutation_rate : float
            Per-gene probability of mutation.
        mutation_sigma : float
            Mutation magnitude as a fraction of (upper - lower) bound, gaussian.
        crossover_rate : float
            Probability of doing crossover (vs cloning a parent).
        init_jitter : float
            Initial population spread, as fraction of bound width around seed.
        """
        self.objective_fn = objective_fn
        # optional batch evaluator: maps a list of parameter vectors -> list of
        # fitnesses, evaluated in parallel (ParallelObjective.evaluate_batch).
        # When set, each generation's population is scored on the pool instead of
        # the serial objective_fn loop.
        self.batch_objective_fn = batch_objective_fn
        self.initial_params = initial_params
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_n = max(1, int(round(elite_frac * population_size)))
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.init_jitter = init_jitter
        self.output_dir = output_dir or "outputs/calibration"
        self.phase_name = phase_name
        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Param helpers
    # -------------------------------------------------------------------------
    def _make_reconstructor(self):
        # Detect param class: TurningParameters has `left_art`, FlowParameters
        # has `flows` only, DemandParameters has neither
        if hasattr(self.initial_params, "left_art"):
            from demand_calibration.turning_parameters import TurningParameters
            template = self.initial_params
            def from_vec(vec):
                return TurningParameters.from_vector(vec, template)
        elif hasattr(self.initial_params, "flows"):
            from demand_calibration.flow_parameters import FlowParameters
            template = self.initial_params
            def from_vec(vec):
                return FlowParameters.from_vector(vec, template)
        else:
            from demand_calibration.demand_parameters import DemandParameters
            cal_hours = self.initial_params.calibration_hours
            def from_vec(vec):
                return DemandParameters.from_vector(vec, cal_hours)
        return from_vec

    def _bounds(self):
        """Return (lo, hi) arrays matching the parameter vector.

        For TurningParameters, the last 4 entries are turning ratios in [0, 1]
        and need different bounds than the vph values.
        """
        # per-movement-type bounds (TurningParameters: arterial vs cross factors)
        if hasattr(self.initial_params, "vector_bounds"):
            return self.initial_params.vector_bounds()
        v0 = self.initial_params.to_vector()
        if hasattr(self.initial_params, "bounds_lower_factor"):
            lo_f = self.initial_params.bounds_lower_factor
            hi_f = self.initial_params.bounds_upper_factor
            lo = np.maximum(0.0, v0 * lo_f)
            hi = np.maximum(10.0, v0 * hi_f)
        else:
            lo = np.maximum(0.0, v0 * 0.5)
            hi = np.maximum(10.0, v0 * 2.0)

        # Special-case for TurningParameters: last 4 entries are turning ratios
        # bounded by TURN_LOW / TURN_HIGH (see turning_parameters.py)
        if hasattr(self.initial_params, "left_art"):
            from demand_calibration.turning_parameters import (
                TURN_LOW, TURN_HIGH)
            lo[-4:] = TURN_LOW
            hi[-4:] = TURN_HIGH
        return lo, hi

    def _clip(self, vec, lo, hi):
        return np.clip(vec, lo, hi)

    # -------------------------------------------------------------------------
    # GA operators
    # -------------------------------------------------------------------------
    def _init_population(self, lo, hi):
        v0 = self.initial_params.to_vector()
        width = hi - lo
        pop = [v0.copy()]  # always include seed
        while len(pop) < self.population_size:
            jitter = self.rng.normal(0.0, self.init_jitter, size=v0.shape) * width
            pop.append(self._clip(v0 + jitter, lo, hi))
        return np.array(pop)

    def _crossover(self, p1, p2):
        """Uniform crossover."""
        mask = self.rng.random(p1.shape) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, vec, lo, hi):
        width = hi - lo
        mask = self.rng.random(vec.shape) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_sigma, size=vec.shape) * width
        out = vec + mask * noise
        return self._clip(out, lo, hi)

    def _tournament_pick(self, pop, fits, k=3):
        idx = self.rng.integers(0, len(pop), size=k)
        winner = idx[np.argmin(fits[idx])]
        return pop[winner]

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def optimize(self):
        from_vec = self._make_reconstructor()
        lo, hi = self._bounds()

        os.makedirs(self.output_dir, exist_ok=True)
        log = []
        log_path = os.path.join(
            self.output_dir,
            f"ga_log_{self.phase_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )

        def flush_log():
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  GA Optimization — {self.phase_name}")
        print(f"  Pop={self.population_size}, Gens={self.n_generations}, "
              f"Elite={self.elite_n}, Genes={len(lo)}")
        print(f"  Log: {log_path}")
        print(f"{'='*60}\n")

        pop = self._init_population(lo, hi)
        fits = np.full(len(pop), np.inf)

        # Evaluate initial population
        print(f"  Generation 0 (initial population):")
        if self.batch_objective_fn is not None:
            fits[:] = self.batch_objective_fn(list(pop))     # parallel
            for i in range(len(pop)):
                print(f"    indiv {i+1:2d}/{len(pop)}: z={fits[i]:.3f}")
        else:
            for i in range(len(pop)):
                params = from_vec(pop[i])
                params.project_bounds()
                pop[i] = params.to_vector()
                fits[i] = self.objective_fn(params)
                print(f"    indiv {i+1:2d}/{len(pop)}: z={fits[i]:.3f}")

        best_idx = int(np.argmin(fits))
        best_z = float(fits[best_idx])
        best_vec = pop[best_idx].copy()
        print(f"  → Gen 0 best: z={best_z:.3f}")

        log.append({
            "generation": 0,
            "best_z": round(best_z, 3),
            "mean_z": round(float(np.mean(fits)), 3),
            "std_z": round(float(np.std(fits)), 3),
            "best_params": best_vec.tolist(),
        })
        flush_log()

        for gen in range(1, self.n_generations + 1):
            # Sort by fitness ascending
            order = np.argsort(fits)
            elites = pop[order[:self.elite_n]].copy()
            elite_fits = fits[order[:self.elite_n]].copy()

            # Build next generation
            next_pop = list(elites)
            while len(next_pop) < self.population_size:
                p1 = self._tournament_pick(pop, fits)
                p2 = self._tournament_pick(pop, fits)
                if self.rng.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.copy()
                child = self._mutate(child, lo, hi)
                next_pop.append(child)

            next_pop = np.array(next_pop)
            next_fits = np.full(len(next_pop), np.inf)
            # Reuse cached fitness for elites (they are unchanged)
            next_fits[:self.elite_n] = elite_fits

            print(f"\n  Generation {gen}/{self.n_generations}:")
            if self.batch_objective_fn is not None:
                idx = list(range(self.elite_n, len(next_pop)))
                next_fits[self.elite_n:] = self.batch_objective_fn(
                    [next_pop[i] for i in idx])              # parallel (non-elites)
                for i in idx:
                    print(f"    indiv {i+1:2d}/{len(next_pop)}: z={next_fits[i]:.3f}")
            else:
                for i in range(self.elite_n, len(next_pop)):
                    params = from_vec(next_pop[i])
                    params.project_bounds()
                    next_pop[i] = params.to_vector()
                    next_fits[i] = self.objective_fn(params)
                    print(f"    indiv {i+1:2d}/{len(next_pop)}: z={next_fits[i]:.3f}")

            pop = next_pop
            fits = next_fits

            gen_best_idx = int(np.argmin(fits))
            if fits[gen_best_idx] < best_z:
                best_z = float(fits[gen_best_idx])
                best_vec = pop[gen_best_idx].copy()

            print(f"  → Gen {gen} best={fits[gen_best_idx]:.3f}  "
                  f"overall_best={best_z:.3f}  mean={np.mean(fits):.3f}")

            log.append({
                "generation": gen,
                "best_z": round(best_z, 3),
                "gen_best_z": round(float(fits[gen_best_idx]), 3),
                "mean_z": round(float(np.mean(fits)), 3),
                "std_z": round(float(np.std(fits)), 3),
                "best_params": best_vec.tolist(),
            })
            flush_log()

        print(f"\n  Log saved: {log_path}")

        best_params = from_vec(best_vec)
        best_params.project_bounds()
        return best_params, best_z, log
