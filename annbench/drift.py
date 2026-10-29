"""Part 4's data generator: Gaussian clusters, old data, new data and queries, with an optional shift.

Everything random is drawn once from one seed, in a fixed order, and the shift is only added afterwards. So the vectors at
shift 0 are exactly the unshifted vectors, and the data at the other shifts differ from them by a single translation:
the size of the shift is the only thing that changes between settings.
"""
from dataclasses import dataclass

import numpy as np

SEED = 4
DIM = 64
CLUSTERS = 100
SIGMA = 1.0          # standard deviation of a cluster along every dimension
CENTRE_SPREAD = 2.0  # standard deviation of the cluster centres along every dimension, in units of SIGMA


@dataclass
class DriftData:
    old: np.ndarray            # (n_old, d) float32: what the indexes are built on
    new: np.ndarray            # (n_new, d) float32: added later; the same clusters, then shifted
    queries: np.ndarray        # (nq, d) float32: drawn like `new`, then shifted
    direction: np.ndarray      # (d,) unit vector the new data and the queries move along
    unit: float                # the length of one unit of shift: a typical vector's distance to its own cluster centre
    shift: float               # the shift in these units

    @property
    def shift_length(self) -> float:
        return self.shift * self.unit


def make_drift_data(shift: float, n_old: int = 200_000, n_new: int = 100_000, n_queries: int = 2000, d: int = DIM,
                    clusters: int = CLUSTERS, seed: int = SEED) -> DriftData:
    """Old vectors, new vectors and queries from `clusters` equally likely Gaussian clusters; the last two moved by `shift`.

    `shift` is measured in units of sigma * sqrt(d), the typical distance of a vector to its own cluster centre
    (the expected norm of a Gaussian vector is just under that). The same unit vector is used at every shift.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, CENTRE_SPREAD * SIGMA, size=(clusters, d))

    def draw(n: int) -> np.ndarray:
        return centres[rng.integers(0, clusters, n)] + rng.normal(0.0, SIGMA, size=(n, d))

    old, new, queries = draw(n_old), draw(n_new), draw(n_queries)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    unit = SIGMA * np.sqrt(d)
    move = shift * unit * direction
    return DriftData(old.astype(np.float32), (new + move).astype(np.float32), (queries + move).astype(np.float32),
                     direction, unit, shift)
