import abc
from typing import Generic

import equinox as eqx

from inferix.custom_types import Y, Aux, SamplerState

class AbstractStepSampler(eqx.Module, Generic[Y, SamplerState, Aux]):
    """
    Abstract base class for all Nested Sampling algorithms.
    """

    @abc.abstractmethod
    def init(self, *args, **kwargs) -> SamplerState:
        """Initialize the sampler's internal state."""

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> tuple[Y, SamplerState, Aux]:
        """Perform one Nested sampling step."""