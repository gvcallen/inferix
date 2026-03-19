"""
Inferix: A unified interface for probabilistic inference in JAX + Equinox.
"""

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# --- MCMC API ---
from inferix.mcmc import (
    AbstractMCMCSampler,
    mcmc_sample,
)

# --- Nested Sampling API ---
from .nested import (
    AbstractHostHypercubeNS,
    AbstractHostPhysicalNS,
    AbstractHypercubeNS,
    AbstractNestedSampler,
    AbstractPhysicalNS,
    nested_sample,
)

# --- Implementation ---
from inferix.samplers.nss import NSS
from inferix.samplers.polychord import PolyChord
from inferix.samplers.nuts import NUTS

__all__ = [
    # Core Runners
    "mcmc_sample",
    "nested_sample",
    
    # Concrete Samplers
    "NUTS",
    "NSS",
    "PolyChord",
    
    # Data Structures
    "Result",
    
    # Abstract Base Classes (Traits for users who want to write their own samplers)
    "AbstractMCMCSampler",
    "AbstractNestedSampler",
    "AbstractPhysicalNS",
    "AbstractHypercubeNS",
    "AbstractHostHypercubeNS",
    "AbstractHostPhysicalNS",
]