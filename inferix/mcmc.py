import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree

from inferix.custom_types import Y, Aux, SamplerState
from inferix.result import Result
from inferix.base import AbstractStepSampler

class AbstractMCMCSampler(AbstractStepSampler):
    """
    Abstract base class for all MCMC transition kernels.
    
    A sampler purely defines the mathematics of proposing and accepting 
    a single transition. It operates on a single point in the parameter space.
    """

    @abc.abstractmethod
    def init(
        self,
        log_prob_fn: Callable,
        y: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
    ) -> SamplerState:
        """
        Initialize the sampler's internal state.

        Arguments:
        - `log_prob_fn`: Callable taking `(y, args)` and returning a scalar log-probability.
        - `y`: The initial parameter guess (PyTree).
        - `args`: Arguments passed to `log_prob_fn(y, args)`.
        - `key`: A JAX PRNGKey for stochastic initialization.
        - `options`: Sampler-specific runtime configuration.

        Returns:
        - A PyTree representing the initial algorithmic state (e.g., identity mass matrix, 
          initial momentum, or step size).
        """

    @abc.abstractmethod
    def step(
        self,
        log_prob_fn: Callable,
        y: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
        state: SamplerState,
    ) -> tuple[Y, SamplerState, Aux]:
        """
        Perform one Markov transition.

        Arguments:
        - `log_prob_fn`: The target log-probability function.
        - `y`: The current parameter state in the Markov chain.
        - `args`: Arguments passed to `log_prob_fn(y, args)`.
        - `key`: A JAX PRNGKey for the stochastic proposal and acceptance step.
        - `options`: Sampler-specific runtime configuration.
        - `state`: The current algorithmic state.

        Returns:
        - A 3-tuple containing:
          1. `new_y`: The next parameter state (may be identical to `y` if rejected).
          2. `new_state`: The updated sampler state.
          3. `aux`: Auxiliary data from the transition (e.g., acceptance probability, 
             number of leapfrog steps taken, or energy error).
        """


@eqx.filter_jit
def mcmc_sample(
    log_prob_fn: Callable,
    sampler: AbstractMCMCSampler,
    y0: Y,
    key: PRNGKeyArray,
    args: PyTree = None,
    options: dict[str, Any] | None = None,
    *,
    num_samples: int = 1000,
    num_burnin: int = 500,
) -> Result:
    """
    Execute an MCMC run, separating the burn-in and sampling phases for memory efficiency.
    """
    if options is None:
        options = {}

    # 1. Initialize the sampler state
    init_state = sampler.init(log_prob_fn, y0, args, key, options)

    # 2. Define the core transition step for jax.lax.scan
    def scan_step(carry, _):
        current_y, current_state, current_key = carry
        
        step_key, next_key = jax.random.split(current_key)
        
        new_y, new_state, aux = sampler.step(
            log_prob_fn, current_y, args, step_key, options, current_state
        )
        
        return (new_y, new_state, next_key), (new_y, aux)

    # 3. Execute Burn-in Phase
    # We use jax.lax.scan but discard the stacked outputs (the second return value)
    burnin_carry = (y0, init_state, key)
    
    if num_burnin > 0:
        burnin_carry, _ = jax.lax.scan(
            scan_step,
            burnin_carry,
            xs=None,
            length=num_burnin
        )

    # 4. Execute Sampling Phase
    # We pass the warmed-up carry state in, and this time we keep the stacked outputs
    final_carry, (samples, auxes) = jax.lax.scan(
        scan_step,
        burnin_carry,
        xs=None,
        length=num_samples
    )
    
    _, final_state, _ = final_carry

    return Result(
        samples=samples,
        aux=auxes,
        final_state=final_state,
    )