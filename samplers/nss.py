from collections.abc import Callable
from typing import Any

import blackjax
from jaxtyping import PRNGKeyArray, PyTree

from inferix.custom_types import Aux, SamplerState, Y
from inferix.nested import AbstractPhysicalNS 


class NSS(AbstractPhysicalNS[Y, SamplerState, Aux]):
    """
    Nested Slice Sampler (NSS) using the BlackJAX backend.
    
    This is a Physical Nested Sampler, meaning it operates directly on 
    the target parameter space and expects a log-prior density function.
    """
    
    num_delete: int
    num_inner_steps: int
    
    def _build_kernel(
        self, 
        likelihood_fn: Callable, 
        prior_fn: Callable, 
        args: PyTree
    ):
        """
        Internal helper to construct the BlackJAX NSS kernel.
        Closes over the args so BlackJAX only sees the active PyTree `y`.
        """
        def logprior(y):
            return prior_fn(y, args)
            
        def loglikelihood(y):
            return likelihood_fn(y, args)
            
        return blackjax.nss(
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner_steps
        )

    def init(
        self,
        likelihood_fn: Callable,
        prior_fn: Callable,
        y_live: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
    ) -> SamplerState:
        
        kernel = self._build_kernel(likelihood_fn, prior_fn, args)
        
        # BlackJAX evaluates the initial likelihoods and priors for the entire 
        # y_live population and sets up the initial logZ tracking.
        state = kernel.init(y_live)
        
        return state

    def step(
        self,
        likelihood_fn: Callable,
        prior_fn: Callable,
        y_live: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
        state: SamplerState,
    ) -> tuple[Y, SamplerState, Aux]:
        
        kernel = self._build_kernel(likelihood_fn, prior_fn, args)
        
        # BlackJAX NSS step identifies the worst `num_delete` particles, 
        # runs slice sampling to generate replacements, and updates logZ.
        new_state, dead_info = kernel.step(key, state)
        
        # new_state.particles contains the updated population of live points.
        # dead_info contains the removed particles, their likelihoods, and log-volumes.
        return new_state.particles, new_state, dead_info