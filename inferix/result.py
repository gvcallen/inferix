from typing import Any

import equinox as eqx
from jaxtyping import Array, Bool, PyTree, Scalar

from inferix.custom_types import SamplerState

class Result(eqx.Module):
    """The result of a sampling run."""
    samples: PyTree[Array]          # The stacked trajectory of accepted posterior samples
    aux: PyTree[Array]              # Stacked auxiliary data (e.g., acceptance probabilities)
    final_state: SamplerState       # The final algorithmic state (useful for resuming chains)
    logZ: Scalar | None = None      # The final log-evidence estimate
    logZ_err: Scalar | None = None  # The estimated error on logZ
    stats: dict[str, Any]           # Sampler-specific info (acceptance rates, n_evals, etc.)
    converged: Bool[Array, ""]      # Whether it hit logZ_convergence before max_iters