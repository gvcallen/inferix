from typing import Any

import equinox as eqx
from jaxtyping import Array, Bool, PyTree, Scalar

from inferix.custom_types import SamplerState

class Result(eqx.Module):
    """The result of a sampling run."""
    samples: PyTree[Array]                                                      # The stacked trajectory of accepted posterior samples
    final_state: SamplerState                                                   # The final algorithmic state (useful for resuming chains)
    aux: PyTree[Array] | None = None                                            # Stacked auxiliary data (e.g., acceptance probabilities)
    stats: dict[str, Any] | None = eqx.field(default_factory=dict)              # Sampler-specific info (acceptance rates, n_evals, etc.)
    converged: Bool[Array, ""] | None = None                                    # Whether the algorithm converged (for nested samplers)
    logZ: Scalar | None = None                                                  # The final log-evidence estimate
    logZ_err: Scalar | None = None                                              # The estimated error on logZ