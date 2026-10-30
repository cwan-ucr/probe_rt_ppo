"""
Define the demand parameter space for calibration optimization.
"""

import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy


# Default hourly volumes (same as generate_daily_demand_with_probe.py)
DEFAULT_HOURLY = {
    0: 80, 1: 60, 2: 40, 3: 40, 4: 80, 5: 300, 6: 700,
    7: 1200, 8: 1600, 9: 1300, 10: 1000, 11: 800, 12: 750,
    13: 780, 14: 900, 15: 1100, 16: 1300, 17: 1600,
    18: 1500, 19: 1100, 20: 700, 21: 450, 22: 250, 23: 120,
}

# Parameter bounds
VOLUME_BOUNDS = (50, 3000)       # vehicles/hour
ARTERIAL_FACTOR_BOUNDS = (1.0, 5.0)


@dataclass
class DemandParameters:
    """Calibration parameters for traffic demand generation.

    Phase 1 optimizes: hourly_volumes for calibration_hours
    Phase 2 optimizes: arterial_factor
    """
    hourly_volumes: dict = field(default_factory=lambda: DEFAULT_HOURLY.copy())
    arterial_factor: float = 2.0
    calibration_hours: tuple = (7, 13)  # morning peak window

    def get_calibration_volumes(self):
        """Return only the volumes being calibrated."""
        start, end = self.calibration_hours
        return {h: self.hourly_volumes[h] for h in range(start, end)}

    def to_vector(self):
        """Flatten calibration parameters to a 1D array.

        Layout: [vol_h7, vol_h8, ..., vol_h12, arterial_factor]
        """
        start, end = self.calibration_hours
        vols = [self.hourly_volumes[h] for h in range(start, end)]
        return np.array(vols + [self.arterial_factor], dtype=float)

    @classmethod
    def from_vector(cls, vec, calibration_hours=(7, 13)):
        """Reconstruct from optimizer output vector."""
        start, end = calibration_hours
        n_hours = end - start
        params = cls(calibration_hours=calibration_hours)

        for i, h in enumerate(range(start, end)):
            params.hourly_volumes[h] = int(np.clip(
                vec[i], VOLUME_BOUNDS[0], VOLUME_BOUNDS[1]
            ))

        params.arterial_factor = float(np.clip(
            vec[n_hours], ARTERIAL_FACTOR_BOUNDS[0], ARTERIAL_FACTOR_BOUNDS[1]
        ))
        return params

    def perturb(self, epsilon, direction):
        """Create a perturbed copy: params + epsilon * direction.

        Parameters
        ----------
        epsilon : float
            Perturbation magnitude
        direction : np.ndarray
            Perturbation direction (same shape as to_vector())

        Returns
        -------
        DemandParameters
        """
        vec = self.to_vector() + epsilon * direction
        return DemandParameters.from_vector(vec, self.calibration_hours)

    def project_bounds(self):
        """Clip all parameters to feasible bounds."""
        for h in self.hourly_volumes:
            self.hourly_volumes[h] = int(np.clip(
                self.hourly_volumes[h], VOLUME_BOUNDS[0], VOLUME_BOUNDS[1]
            ))
        self.arterial_factor = float(np.clip(
            self.arterial_factor,
            ARTERIAL_FACTOR_BOUNDS[0],
            ARTERIAL_FACTOR_BOUNDS[1],
        ))

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        start, end = self.calibration_hours
        vols = {h: self.hourly_volumes[h] for h in range(start, end)}
        return (f"DemandParameters(vols={vols}, "
                f"arterial_factor={self.arterial_factor:.2f})")
