import abc
from typing import Generic

import equinox as eqx

from jaxtyping import Bool, Array
from inferix.custom_types import Out, Y, Aux, SamplerState
from inferix.result import Result, RESULTS

class AbstractSampler(eqx.Module, Generic[Y, Out, Aux, SamplerState]):
    """Abstract base class for all samplers."""

    @abc.abstractmethod
    def init(self, *args, **kwargs) -> SamplerState:
        """Perform all initial computation needed to initialise the sampler state."""

    @abc.abstractmethod
    def step(self,) -> tuple[Y, SamplerState, Aux]:
        """Perform one step of the sampling."""


class AbstractIterativeSampler(AbstractSampler, Generic[Y, Out, Aux, SamplerState]):
    """Abstract base class for all iterative solvers."""

    @abc.abstractmethod
    def terminate(self, *args, **kwargs) -> tuple[Bool[Array, ""], RESULTS]:
        """Determine whether or not to stop the iterative sampling."""

class AbstractHostSampler(eqx.Module):
    """Abstract base class for all host-native solvers."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Result:
        """ Run the full sampling and return the final result. """
