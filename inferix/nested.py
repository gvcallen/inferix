import abc
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from inferix.custom_types import Y, Aux, SamplerState
from inferix.result import Result
from inferix.base import AbstractStepSampler

class NSInfo(eqx.Module):
    """
    Standardized Auxiliary output for a single Nested Sampling step.
    This guarantees the runner always knows where the physical parameters are.
    """
    particles: PyTree[Array]   # The parameter coordinates (theta or u)
    loglikelihood: Array       # L(theta)
    loglikelihood_birth: Array

class AbstractNestedSampler(AbstractStepSampler):
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
    def run(self, log_likelihood_fn: Callable, prior_transform_fn: Callable, ndims: int, args: PyTree) -> Result:
        """Executes the host-driven algorithm and returns the standard solution."""


class AbstractHostPhysicalNS(AbstractHostNestedSampler):
    """
    Trait for Nested Samplers that operate in the physical space
    but control their own host-side execution loop.
    """
    @abc.abstractmethod
    def run(self, log_likelihood_fn: Callable, log_prior_fn: Callable, ndims: int, args: PyTree) -> Result:
        """Executes the host-driven algorithm and returns the standard solution."""


def _uniform_log_prior(u: PyTree, args: PyTree) -> Scalar:
    """A log-prior that strictly bounds the sampler to the [0, 1] hypercube."""
    leaves, _ = jtu.tree_flatten(u)
    # Check if all elements in the PyTree are between 0 and 1
    in_bounds = jnp.all(jnp.array([jnp.all((x >= 0.0) & (x <= 1.0)) for x in leaves]))
    return jnp.where(in_bounds, 0.0, -jnp.inf)


@eqx.filter_jit
def _device_loop(
    log_likelihood_fn, prior_func, sampler, y_live, args, key, options, logZ_convergence, max_iters
):
    """The blazing-fast, strictly-compiled JAX execution engine for native JAX samplers."""
    init_state = sampler.init(log_likelihood_fn, prior_func, y_live, args, key, options)
    
    dummy_key, _ = jax.random.split(key)
    _, _, dummy_aux = eqx.filter_eval_shape(
        sampler.step, log_likelihood_fn, prior_func, y_live, args, dummy_key, options, init_state
    )

    buffer_init = jtu.tree_map(lambda x: jnp.zeros((max_iters, *x.shape), dtype=x.dtype), dummy_aux)
    init_carry = (jnp.array(0), init_state, y_live, key, buffer_init)

    def cond_fun(carry):
        step_idx, state, _, _, _ = carry
        # Stop if converged or we hit the pre-allocated max_iters buffer size
        not_converged = (state.logZ_live - state.logZ) >= logZ_convergence
        under_max_steps = step_idx < max_iters
        return not_converged & under_max_steps

    def body_fun(carry):
        step_idx, state, current_y_live, current_key, buffer = carry
        step_key, next_key = jax.random.split(current_key)
        
        new_y_live, new_state, dead_info = sampler.step(
            log_likelihood_fn, prior_func, current_y_live, args, step_key, options, state
        )
        
        # Insert the dead point into the buffer at the current index
        new_buffer = jtu.tree_map(lambda b, d: b.at[step_idx].set(d), buffer, dead_info)
        return step_idx + 1, new_state, new_y_live, next_key, new_buffer

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    final_steps, final_state, _, _, final_buffer = final_carry
    converged = (final_state.logZ_live - final_state.logZ) < logZ_convergence

    return final_buffer, final_state, final_steps, converged


def nested_sample(
    log_likelihood_fn: Callable,
    sampler: eqx.Module,  # Accepts any valid sampler trait
    key: PRNGKeyArray,
    args: PyTree = None,
    options: dict[str, Any] | None = None,
    *,
    log_prior_fn: Callable | None = None,        
    prior_sample_fn: Callable[[PRNGKeyArray, PyTree], Y] | None = None,
    prior_transform_fn: Callable | None = None,  
    y_live: Y | None = None,
    nlive: int | None = None,
    ndims: int | None = None,                    
    logZ_convergence: float = 1e-3, 
    max_iters: int = 100000,        
) -> Result: # type: ignore
    """
    Execute a Nested Sampling run. Unifies JAX-native and Host-native algorithms.
    """
    if options is None:
        options = {}

    # --- 1. HOST-DRIVEN SAMPLERS (e.g., PolyChord) ---
    if isinstance(sampler, AbstractHostHypercubeNS):
        if prior_transform_fn is None or ndims is None:
            raise ValueError(f"{sampler.__class__.__name__} requires `prior_transform_fn` and `ndims`.")
        
        # Bypass the JAX loop entirely and yield to the host algorithm
        return sampler.run(log_likelihood_fn, prior_transform_fn, ndims, args)

    elif isinstance(sampler, AbstractHostPhysicalNS):
        if log_prior_fn is None or ndims is None:
            raise ValueError(f"{sampler.__class__.__name__} requires `log_prior_fn` and `ndims`.")
            
        return sampler.run(log_likelihood_fn, log_prior_fn, ndims, args)

    # --- 2. JAX-NATIVE SAMPLERS (e.g., BlackJAX NSS) ---
    is_hypercube_run = False
    
    if isinstance(sampler, AbstractPhysicalNS):
        if prior_transform_fn is not None:
            # Reparameterization Trick: Run physical sampler in the hypercube
            is_hypercube_run = True
            prior_func = _uniform_log_prior
            likelihood_func = lambda u, args: log_likelihood_fn(prior_transform_fn(u, args), args)
            
            if y_live is None:
                if nlive is None or ndims is None:
                    raise ValueError("To auto-initialize via prior_transform, provide `nlive` and `ndims`.")
                init_key, key = jax.random.split(key)
                y_live = jax.random.uniform(init_key, shape=(nlive, ndims))
                
        elif log_prior_fn is not None:
            # Standard physical run
            prior_func = log_prior_fn
            likelihood_func = log_likelihood_fn
            
            if y_live is None:
                if nlive is None or prior_sample_fn is None:
                    raise ValueError("To auto-initialize a Physical NS, provide `nlive` and `prior_sample_fn`.")
                init_key, key = jax.random.split(key)
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
            if nlive is None or ndims is None:
                raise ValueError("To auto-initialize a Hypercube NS, provide `nlive` and `ndims`.")
            init_key, key = jax.random.split(key)
            y_live = jax.random.uniform(init_key, shape=(nlive, ndims))
            
    else:
        raise TypeError("Sampler must inherit from a valid AbstractNestedSampler trait.")

    # --- 3. EXECUTE JAX LOOP ---
    final_buffer, final_state, final_steps, converged = _device_loop(
        likelihood_func, prior_func, sampler, y_live, args, key, options, logZ_convergence, max_iters
    )

    # --- 4. POST-PROCESSING (Auto-Transform Hypercube to Physical Space) ---
    if is_hypercube_run and prior_transform_fn is not None:
        
        @jax.jit
        def _transform_buffer(buffer_to_transform: NSInfo, current_args: PyTree):
            # 1. Vectorize the user's prior transform function over the 0th axis (max_iters).
            #    in_axes=(0, None) means: batch over `u`, but pass `args` unchanged.
            batched_transform = jax.vmap(prior_transform_fn, in_axes=(0, None))
            
            # 2. Apply the transform to the stacked [0, 1] hypercube particles.
            physical_particles = batched_transform(buffer_to_transform.particles, current_args)
            
            # 3. Surgically replace the `particles` leaf in the NSInfo PyTree
            return eqx.tree_at(
                lambda b: b.particles, 
                buffer_to_transform, 
                physical_particles
            )

        # Apply the transformation to the final accumulated buffer
        final_buffer = _transform_buffer(final_buffer, args)

    return Result(
        samples=final_buffer, 
        logZ=final_state.logZ,
        logZ_err=final_state.logZ_err if hasattr(final_state, 'logZ_err') else jnp.nan,
        converged=converged,
        final_state=final_state
    )