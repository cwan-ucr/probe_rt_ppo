"""
SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer
for demand calibration.

Reference: Spall, J.C. (1998). Implementation of the Simultaneous
Perturbation Algorithm for Stochastic Optimization.
"""

import numpy as np
import json
import os
from datetime import datetime


class SPSAOptimizer:
    """SPSA optimizer for traffic demand calibration.

    Each iteration requires only 2 function evaluations regardless of
    parameter dimension, making it efficient for simulation-based optimization.
    """

    def __init__(self, objective_fn, initial_params,
                 a=100.0, c=5.0, A=10, alpha=0.602, gamma=0.101,
                 max_iterations=30, convergence_threshold=0.5,
                 output_dir=None, phase_name=""):
        """
        Parameters
        ----------
        objective_fn : callable
            Function that takes DemandParameters and returns float (objective value)
        initial_params : DemandParameters
            Starting point
        a, c, A, alpha, gamma : float
            SPSA gain sequence parameters (Spall 1998 standard)
        max_iterations : int
            Maximum number of SPSA iterations
        convergence_threshold : float
            Stop if |Z_{k+1} - Z_k| < threshold for 3 consecutive iterations
        output_dir : str
            Directory for logging
        phase_name : str
            Label for this optimization phase
        """
        self.objective_fn = objective_fn
        self.initial_params = initial_params
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.output_dir = output_dir or "outputs/calibration"
        self.phase_name = phase_name

    def _gain_a(self, k):
        return self.a / (k + 1 + self.A) ** self.alpha

    def _gain_c(self, k):
        return self.c / (k + 1) ** self.gamma

    def optimize(self):
        """Run SPSA optimization loop.

        Returns
        -------
        best_params : initial_params type (DemandParameters or FlowParameters)
            Best parameters found
        best_z : float
            Best objective value
        log : list[dict]
            Iteration log
        """
        # Generic param reconstructor: uses initial_params type (duck-typed).
        # FlowParameters has `flows` attribute, DemandParameters has `hourly_volumes`.
        is_flow = hasattr(self.initial_params, "flows")

        if is_flow:
            from .flow_parameters import FlowParameters
            def params_from_vector(vec):
                return FlowParameters.from_vector(vec, self.initial_params)
        else:
            from .demand_parameters import DemandParameters
            def params_from_vector(vec):
                return DemandParameters.from_vector(vec, self.initial_params.calibration_hours)

        os.makedirs(self.output_dir, exist_ok=True)
        log = []

        theta = self.initial_params.to_vector()
        p = len(theta)
        cal_hours = self.initial_params.calibration_hours

        best_z = float("inf")
        best_theta = theta.copy()
        consecutive_small = 0
        prev_z = None

        print(f"\n{'='*60}")
        print(f"  SPSA Optimization — {self.phase_name}")
        print(f"  Parameters: {p}, Max iterations: {self.max_iterations}")
        print(f"{'='*60}\n")

        for k in range(self.max_iterations):
            a_k = self._gain_a(k)
            c_k = self._gain_c(k)

            # Random perturbation from symmetric Bernoulli
            delta = np.random.choice([-1, 1], size=p).astype(float)

            # Evaluate at theta + c_k*delta and theta - c_k*delta
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            params_plus = params_from_vector(theta_plus)
            params_minus = params_from_vector(theta_minus)

            print(f"  Iteration {k+1}/{self.max_iterations}: ", end="", flush=True)

            z_plus = self.objective_fn(params_plus)
            z_minus = self.objective_fn(params_minus)

            # Gradient estimate
            g_hat = (z_plus - z_minus) / (2.0 * c_k * delta)

            # Update
            theta = theta - a_k * g_hat

            # Project to bounds
            current_params = params_from_vector(theta)
            current_params.project_bounds()
            theta = current_params.to_vector()

            # Evaluate current point
            z_current = min(z_plus, z_minus)

            if z_current < best_z:
                best_z = z_current
                best_theta = theta.copy()

            entry = {
                "iteration": k + 1,
                "z_plus": round(z_plus, 3),
                "z_minus": round(z_minus, 3),
                "z_best": round(best_z, 3),
                "a_k": round(a_k, 4),
                "c_k": round(c_k, 4),
                "params": current_params.to_vector().tolist(),
            }
            log.append(entry)

            print(f"Z+={z_plus:.2f}  Z-={z_minus:.2f}  "
                  f"best={best_z:.2f}  a_k={a_k:.3f}  c_k={c_k:.3f}")

            # Convergence check
            if prev_z is not None:
                if abs(z_current - prev_z) < self.convergence_threshold:
                    consecutive_small += 1
                else:
                    consecutive_small = 0

                if consecutive_small >= 3:
                    print(f"\n  Converged after {k+1} iterations "
                          f"(3 consecutive Δ < {self.convergence_threshold})")
                    break

            prev_z = z_current

        # Save log
        log_path = os.path.join(
            self.output_dir,
            f"spsa_log_{self.phase_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n  Log saved: {log_path}")

        best_params = params_from_vector(best_theta)
        return best_params, best_z, log
