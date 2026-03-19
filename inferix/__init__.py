"""
Inferix: A unified interface for probabilistic inference in JAX + Equinox.
"""

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# --- Core API ---
from inferix.result import Result
from inferix.base import AbstractSampler, AbstractIterativeSampler

# --- MCMC API ---
from inferix.mcmc import (
    AbstractMCMCSampler,
    mcmc_sample,
)

# --- Nested Sampling API ---
from inferix.nested import (
    AbstractHostHypercubeNS,
    AbstractHostPhysicalNS,
    AbstractHypercubeNS,
    AbstractNestedSampler,
    AbstractPhysicalNS,
    nested_sample,
)

# --- Implementation ---
from inferix.samplers import *
from inferix import samplers

__all__ = [
    "AbstractSampler",
    "AbstractIterativeSampler",
    "Result",
    "AbstractMCMCSampler",
    "mcmc_sample",
    "AbstractHostHypercubeNS",
    "AbstractHostPhysicalNS",
    "AbstractHypercubeNS",
    "AbstractNestedSampler",
    "AbstractPhysicalNS",
    "nested_sample",
]

__all__.extend(samplers.__all__)