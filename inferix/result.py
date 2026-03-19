from typing import Any

import equinox as eqx
import equinox.internal as eqxi
from jaxtyping import Array, Bool, PyTree, Scalar

from inferix.custom_types import SamplerState

class RESULTS(eqxi.Enumeration):
    successful = ""
    max_steps_reached = "max_steps_reached"
    # Add other failure states here as needed

class Result(eqx.Module):
    """The result of a sampling run."""
    samples: PyTree[Array]                                                      # The stacked trajectory of samples (Batched PyTree)
    final_state: SamplerState | None                                            # The final algorithmic state (None for PolyChord)
    aux: PyTree[Array] | None = None                                            # Stacked auxiliary data
    stats: dict[str, Any] | None = eqx.field(default_factory=dict)              # Sampler-specific info (e.g. num_steps)
    converged: Bool[Array, ""] | None = None                                    # Whether the algorithm converged
    logZ: Scalar | None = None                                                  # The final log-evidence estimate
    logZ_err: Scalar | None = None                                              # The estimated error on logZ
    result: RESULTS = RESULTS.successful