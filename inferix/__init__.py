"""
Inferix: A unified interface for probabilistic inference in JAX + Equinox.
"""

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# --- Core API ---
from inferix.result import Result, RESULTS
from inferix.base import AbstractSampler, AbstractIterativeSampler

# --- MCMC API ---
from inferix.mcmc import (
    AbstractMCMCSampler,
    mcmc,
)

# --- Nested Sampling API ---
from inferix.nested import (
    AbstractHypercubeNestedSampler,
    AbstractNestedSampler,
    AbstractPhysicalNestedSampler,
    nested,
)

# --- Implementation ---
from inferix.samplers import *
from inferix import samplers

__all__ = [
    "AbstractSampler",
    "AbstractIterativeSampler",
    "Result",
    "RESULTS",
    "AbstractMCMCSampler",
    "mcmc",
    "AbstractHostHypercubeNestedSampler",
    "AbstractHostPhysicalNestedSampler",
    "AbstractHypercubeNestedSampler",
    "AbstractNestedSampler",
    "AbstractPhysicalNestedSampler",
    "nested",
]

__all__.extend(samplers.__all__)