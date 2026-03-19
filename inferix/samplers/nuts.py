from collections.abc import Callable
from typing import Any, Optional

import blackjax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, PyTree

from inferix.custom_types import Aux, SamplerState, Y
from inferix.mcmc import AbstractMCMCSampler 

class NUTS(AbstractMCMCSampler[Y, SamplerState, Aux]):
    """
    The No-U-Turn Sampler (NUTS) using the BlackJAX backend.
    
    This kernel automatically closes over the target arguments and 
    defaults the inverse mass matrix to identity if not provided.
    """
    
    step_size: float
    inverse_mass_matrix: Optional[PyTree] = None
    max_num_doublings: int = 10
    divergence_threshold: int = 1000
    
    def _build_kernel(self, log_prob_fn: Callable, args: PyTree, y: Y):
        """
        Internal helper to construct the BlackJAX kernel dynamically.
        Because BlackJAX creates pure JAX functions, calling this inside 
        `init` and `step` has zero runtime overhead under `jax.jit`.
        """
        # 1. Close over the args so BlackJAX only sees `y`
        def logdensity(x):
            return log_prob_fn(x, args)
        
        # 2. Auto-initialize the mass matrix to an identity PyTree if missing
        inv_mass = self.inverse_mass_matrix
        if inv_mass is None:
            inv_mass = jtu.tree_map(lambda leaf: jnp.ones_like(leaf), y)
            
        # 3. Return the BlackJAX kernel tuple (contains .init and .step)
        return blackjax.nuts(
            logdensity_fn=logdensity,
            step_size=self.step_size,
            inverse_mass_matrix=inv_mass,
            max_num_doublings=self.max_num_doublings,
            divergence_threshold=self.divergence_threshold,
        )

    def init(
        self,
        log_prob_fn: Callable,
        y: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
    ) -> SamplerState:
        
        kernel = self._build_kernel(log_prob_fn, args, y)
        
        # BlackJAX's init function evaluates the initial log probability 
        # and gradient, returning a structured HMCState.
        state = kernel.init(y)
        
        return state

    def step(
        self,
        log_prob_fn: Callable,
        y: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
        state: SamplerState,
    ) -> tuple[Y, SamplerState, Aux]:
        
        kernel = self._build_kernel(log_prob_fn, args, y)
        
        # BlackJAX's step function takes the PRNGKey and the current HMCState.
        # It returns the new HMCState and an HMCInfo tuple (diagnostics).
        new_state, info = kernel.step(key, state)
        
        # new_state.position contains the updated parameter PyTree `y`
        return new_state.position, new_state, info