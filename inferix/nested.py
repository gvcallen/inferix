import abc
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from inferix.custom_types import Y, Aux, SamplerState
from inferix.result import Result
from inferix.base import AbstractIterativeSampler

class NSInfo(eqx.Module):
    """
    Standardized Auxiliary output for a single Nested Sampling step.
    This guarantees the runner always knows where the physical parameters are.
    """
    particles: PyTree[Array]   # The parameter coordinates (theta or u)
    loglikelihood: Array       # L(theta)
    loglikelihood_birth: Array

class AbstractNestedSampler(AbstractIterativeSampler):
    """
    Abstract base class for all Nested Sampling algorithms.
    """
    @abc.abstractmethod
    def init(
        self,
        likelihood_fn: Callable,
        prior_fn: Callable,
        y_live: Y,
        args: PyTree,
        key: PRNGKeyArray,
        options: dict[str, Any],
    ) -> SamplerState:
        """Initialize the nested sampler's internal state."""

    @abc.abstractmethod
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
        """Perform one Nested Sampling iteration."""


class AbstractHostNestedSampler(eqx.Module):
    """
    Trait for Nested Samplers that control their own host-side execution loop.
    """
    pass


class AbstractPhysicalNS(AbstractNestedSampler[Y, SamplerState, Aux]):
    """
    Trait for Nested Samplers operating in the target physical parameter space (e.g., BlackJAX NSS).
    The runner will enforce that the user provides a `log_prior_fn`.
    """
    pass


class AbstractHypercubeNS(AbstractNestedSampler[Y, SamplerState, Aux]):
    """
    Trait for Nested Samplers operating in the unit hypercube [0, 1]^d (e.g., PolyChord).
    The runner will enforce that the user provides a `prior_transform_fn`.
    """
    pass


class AbstractHostHypercubeNS(AbstractHostNestedSampler):
    """
    Trait for Nested Samplers that natively explore the unit hypercube 
    but control their own host-side execution loop (e.g., PolyChord C++ binary).
    """
    @abc.abstractmethod
    def run(self, log_likelihood_fn: Callable, prior_transform_fn: Callable, y0: Y, args: PyTree) -> Result:
        """Executes the host-driven algorithm and returns the standard solution."""


class AbstractHostPhysicalNS(AbstractHostNestedSampler):
    """
    Trait for Nested Samplers that operate in the physical space
    but control their own host-side execution loop.
    """
    @abc.abstractmethod
    def run(self, log_likelihood_fn: Callable, log_prior_fn: Callable, y0: Y, args: PyTree) -> Result:
        """Executes the host-driven algorithm and returns the standard solution."""


def _uniform_log_prior(u: PyTree, args: PyTree) -> Scalar:
    leaves, _ = jtu.tree_flatten(u)
    in_bounds = jnp.all(jnp.array([jnp.all((x >= 0.0) & (x <= 1.0)) for x in leaves]))
    return jnp.where(in_bounds, 0.0, -jnp.inf)

@eqx.filter_jit
def _device_loop(
    likelihood_func, prior_func, sampler, y_live, args, key, options, max_iters
):
    init_state = sampler.init(likelihood_func, prior_func, y_live, args, key, options)
    
    dummy_key, _ = jax.random.split(key)
    _, _, dummy_aux = eqx.filter_eval_shape(
        sampler.step, likelihood_func, prior_func, y_live, args, dummy_key, options, init_state
    )

    buffer_init = jtu.tree_map(lambda x: jnp.zeros((max_iters, *x.shape), dtype=x.dtype), dummy_aux)
    init_carry = (jnp.array(0), init_state, y_live, key, buffer_init)

    def cond_fun(carry):
        step_idx, state, _, _, _ = carry
        converged, _ = sampler.terminate(state=state)
        under_max_steps = step_idx < max_iters
        return jnp.logical_not(converged) & under_max_steps

    def body_fun(carry):
        step_idx, state, current_y_live, current_key, buffer = carry
        step_key, next_key = jax.random.split(current_key)
        
        new_y_live, new_state, dead_info = sampler.step(
            likelihood_func, prior_func, current_y_live, args, step_key, options, state
        )
        
        new_buffer = jtu.tree_map(lambda b, d: b.at[step_idx].set(d), buffer, dead_info)
        return step_idx + 1, new_state, new_y_live, next_key, new_buffer

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    final_steps, final_state, _, _, final_buffer = final_carry
    
    final_converged, final_status = sampler.terminate(state=final_state)

    return final_buffer, final_state, final_steps, final_converged, final_status

def nested_sample(
    log_likelihood_fn: Callable,
    key: PRNGKeyArray,
    args: PyTree = None,
    sampler: AbstractNestedSampler | None = None,
    options: dict[str, Any] | None = None,
    *,
    y0: Y | None = None,  
    y_live: Y | None = None, 
    log_prior_fn: Callable | None = None,        
    prior_transform_fn: Callable | None = None,  
    prior_sample_fn: Callable | None = None,
    nlive: int | None = None,
    max_steps: int = 100000,        
) -> Result:
    
    if sampler is None:
        from inferix.samplers.nss import NSS
        sampler = NSS()
    if options is None:
        options = {}

    # --- 1. HOST-DRIVEN SAMPLERS ---
    if isinstance(sampler, AbstractHostHypercubeNS):
        if prior_transform_fn is None:
            raise ValueError(f"{sampler.__class__.__name__} requires `prior_transform_fn`.")
        return sampler.run(log_likelihood_fn, prior_transform_fn, y0, args, nlive=nlive)

    elif isinstance(sampler, AbstractHostPhysicalNS):
        if log_prior_fn is None:
            raise ValueError(f"{sampler.__class__.__name__} requires `log_prior_fn`.")
        return sampler.run(log_likelihood_fn, log_prior_fn, y0, args, nlive=nlive)

    # --- 2. INITIALIZE LIVE POINTS ---
    if y_live is None:
        if y0 is None or nlive is None:
            raise ValueError("Must provide either `y_live` OR both `y0` and `nlive`.")
        
        init_key, key = jax.random.split(key)
        
        # [FIXED BUG]: Robust uncorrelated generation using flatten_util
        flat_y0, reconstruct_fn = jax.flatten_util.ravel_pytree(y0)
        ndims = flat_y0.size
        u_live_flat = jax.random.uniform(init_key, shape=(nlive, ndims))
        
        # This vmap automatically outputs a properly batched PyTree of [0, 1] coordinates
        u_live = jax.vmap(reconstruct_fn)(u_live_flat)

    is_hypercube_run = False
    
    # --- 3. JAX-NATIVE ROUTING ---
    if isinstance(sampler, AbstractPhysicalNS):
        if prior_transform_fn is not None:
            is_hypercube_run = True
            prior_func = _uniform_log_prior
            likelihood_func = lambda u, a: log_likelihood_fn(prior_transform_fn(u, a), a)
            
            if y_live is None:
                y_live = u_live
                
        elif log_prior_fn is not None:
            prior_func = log_prior_fn
            likelihood_func = log_likelihood_fn
            
            if y_live is None:
                if prior_sample_fn is None:
                    raise ValueError("Physical samplers require `prior_sample_fn` to auto-initialize.")
                keys = jax.random.split(init_key, nlive)
                y_live = jax.vmap(prior_sample_fn, in_axes=(0, None))(keys, args)
        else:
            raise ValueError("Physical samplers require either `log_prior_fn` or `prior_transform_fn`.")

    elif isinstance(sampler, AbstractHypercubeNS):
        if prior_transform_fn is None:
            raise ValueError("Hypercube samplers require `prior_transform_fn`.")
        is_hypercube_run = True
        prior_func = prior_transform_fn
        likelihood_func = log_likelihood_fn
        
        if y_live is None:
            y_live = u_live
            
    else:
        raise TypeError("Sampler must inherit from a valid AbstractNestedSampler trait.")

    # --- 4. EXECUTE JAX LOOP ---
    final_buffer, final_state, final_steps, converged, status = _device_loop(
        likelihood_func, prior_func, sampler, y_live, args, key, options, max_steps
    )

    # --- 5. POST-PROCESSING -> Creates a Batch of Caller PyTrees! ---
    if is_hypercube_run and prior_transform_fn is not None:
        
        @jax.jit
        def _transform_buffer(buffer_to_transform: NSInfo, current_args: PyTree):
            batched_transform = jax.vmap(prior_transform_fn, in_axes=(0, None))
            physical_particles = batched_transform(buffer_to_transform.particles, current_args)
            return eqx.tree_at(
                lambda b: b.particles, 
                buffer_to_transform, 
                physical_particles
            )

        final_buffer = _transform_buffer(final_buffer, args)

    return Result(
        samples=final_buffer, 
        final_state=final_state,
        logZ=final_state.logZ if hasattr(final_state, 'logZ') else None,
        logZ_err=final_state.logZ_err if hasattr(final_state, 'logZ_err') else None,
        converged=converged,
        result=status,
        stats={"num_steps": final_steps}
    )