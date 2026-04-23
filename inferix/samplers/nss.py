from collections.abc import Callable
from typing import Any

import blackjax
from jaxtyping import PRNGKeyArray, PyTree, Bool, Array

from inferix.custom_types import Aux, SamplerState, Y
from inferix.nested import AbstractPhysicalNestedSampler, NestedSamplingInfo
from inferix.result import RESULTS

from optimistix import minimise

class NSS(AbstractPhysicalNestedSampler[Y, SamplerState, NestedSamplingInfo]):
    """
    Nested Slice Sampler (NSS) using the BlackJAX backend.
    """
    
    num_delete: int
    num_inner_steps: int
    logZ_convergence: float = 1e-3
    
    def _build_kernel(self, likelihood_fn: Callable, prior_fn: Callable, args: PyTree):
        def logprior(y): return prior_fn(y, args)
        def loglikelihood(y): return likelihood_fn(y, args)
            
        return blackjax.nss(
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner_steps
        )

    def init(self, likelihood_fn, prior_fn, y_live, args, key, options) -> SamplerState:
        kernel = self._build_kernel(likelihood_fn, prior_fn, args)
        return kernel.init(y_live)

    def step(
        self,
        likelihood_fn: Callable,
        prior_fn: Callable,
        y_live: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
        state: SamplerState,
    ) -> tuple[Y, SamplerState, NestedSamplingInfo]: # <-- Explicit return type
        
        kernel = self._build_kernel(likelihood_fn, prior_fn, args)
        
        # BlackJAX returns its own internal NSSInfo object
        new_state, blackjax_dead_info = kernel.step(key, state)
        
        # --- THE TRANSLATION LAYER ---
        # Repackage BlackJAX's specific outputs into our unified API contract
        standardized_aux = NestedSamplingInfo(
            particles=blackjax_dead_info.particles,
            loglikelihood=blackjax_dead_info.loglikelihood,
            loglikelihood_birth=blackjax_dead_info.loglikelihood_birth
        )
        
        return new_state.particles, new_state, standardized_aux
    
    def terminate(
        self, 
        state: SamplerState, 
        **kwargs
    ) -> tuple[Bool[Array, ""], RESULTS]:
        """Determine whether the evidence integral has converged."""
        converged = state.logZ_live - state.logZ < self.logZ_convergence
        return converged, RESULTS.successful