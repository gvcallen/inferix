import abc
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from inferix.custom_types import Y, Aux, SamplerState
from inferix.result import Result
from inferix.base import AbstractIterativeSampler, AbstractHostSampler

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


class AbstractHostHypercubeNS(AbstractHostSampler):
    """
    Trait for Nested Samplers that natively explore the unit hypercube 
    but control their own host-side execution loop (e.g., PolyChord C++ binary).
    """
    @abc.abstractmethod
    def __call__(self, log_likelihood_fn: Callable, prior_transform_fn: Callable, y0: Y, args: PyTree, **kwargs) -> Result:
        """Executes the host-driven algorithm and returns the standard solution."""


class AbstractHostPhysicalNS(AbstractHostSampler):
    """
    Trait for Nested Samplers that operate in the physical space
    but control their own host-side execution loop.
    """
    @abc.abstractmethod
    def __call__(self, log_likelihood_fn: Callable, log_prior_fn: Callable, y0: Y, args: PyTree, **kwargs) -> Result:
        """Executes the host-driven algorithm and returns the standard solution."""


def _uniform_log_prior(u: PyTree, args: PyTree) -> Scalar:
    leaves, _ = jtu.tree_flatten(u)
    in_bounds = jnp.all(jnp.array([jnp.all((x >= 0.0) & (x <= 1.0)) for x in leaves]))
    return jnp.where(in_bounds, 0.0, -jnp.inf)


def _batched_loop(
    likelihood_func, prior_func, sampler, y_live, args, key, options, max_steps, batch_size
):
    """
    Executes inference in chunks of `batch_size`.
    Provides XLA compilation speed while allowing infinite tracking and host-side control.
    """
    init_state = sampler.init(likelihood_func, prior_func, y_live, args, key, options)
    
    dummy_key, _ = jax.random.split(key)
    _, _, dummy_aux = eqx.filter_eval_shape(
        sampler.step, likelihood_func, prior_func, y_live, args, dummy_key, options, init_state
    )

    # 1. JIT compile the core mathematical batch
    # We declare static_batch_size so XLA knows exactly how much memory to pre-allocate.
    @eqx.filter_jit(static_argnames=("static_batch_size",))
    def execute_batch(current_y_live, current_key, state, static_batch_size, dynamic_max_steps):
        
        buffer_init = jtu.tree_map(lambda x: jnp.zeros((static_batch_size, *x.shape), dtype=x.dtype), dummy_aux)
        init_carry = (jnp.array(0), state, current_y_live, current_key, buffer_init)

        def cond_fun(carry):
            step_idx, s, _, _, _ = carry
            converged, _ = sampler.terminate(state=s)
            # Stop if converged OR if we hit the requested step limit for this batch
            return jnp.logical_not(converged) & (step_idx < dynamic_max_steps)

        def body_fun(carry):
            step_idx, s, y, k, buffer = carry
            step_k, next_k = jax.random.split(k)
            
            new_y, new_s, dead_info = sampler.step(
                likelihood_func, prior_func, y, args, step_k, options, s
            )
            
            new_buffer = jtu.tree_map(lambda b, d: b.at[step_idx].set(d), buffer, dead_info)
            return step_idx + 1, new_s, new_y, next_k, new_buffer

        final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        steps_taken, final_state, final_y, final_key, final_buffer = final_carry
        
        final_converged, final_status = sampler.terminate(state=final_state)

        return final_y, final_key, final_state, final_buffer, steps_taken, final_converged, final_status

    # 2. Host CPU Controller Loop
    current_y_live = y_live
    current_key = key
    current_state = init_state
    
    buffer_list = []
    total_steps = 0
    
    while True:
        # Determine how many steps to take in the current batch
        if max_steps is None:
            steps_this_batch = batch_size
        else:
            steps_this_batch = min(batch_size, max_steps - total_steps)
            
        if steps_this_batch <= 0:
            # Reached max steps
            converged, status = sampler.terminate(state=current_state)
            break

        # Execute on GPU/TPU
        current_y_live, current_key, current_state, batch_buffer, steps_taken, converged, status = execute_batch(
            current_y_live, current_key, current_state, 
            static_batch_size=batch_size, 
            dynamic_max_steps=steps_this_batch
        )

        # Cast JAX tracers back to Python ints/bools for host control flow
        steps_taken_int = int(steps_taken)
        total_steps += steps_taken_int
        
        # Slice off any trailing zeros if the sampler converged early within the batch
        if steps_taken_int > 0:
            valid_batch_buffer = jtu.tree_map(lambda x: x[:steps_taken_int], batch_buffer)
            buffer_list.append(valid_batch_buffer)
            
        if bool(converged):
            break

    # 3. Stack all batches back into a single PyTree
    if len(buffer_list) > 0:
        # jnp.concatenate is used because each item is already an array of shape (steps_taken, ...)
        final_buffer = jtu.tree_map(lambda *leaves: jnp.concatenate(leaves, axis=0), *buffer_list)
    else:
        # Edge case: 0 steps were taken
        final_buffer = jtu.tree_map(lambda x: jnp.zeros((0, *x.shape), dtype=x.dtype), dummy_aux)

    return final_buffer, current_state, total_steps, converged, status


def nested_sample(
    log_likelihood_fn: Callable,
    key: PRNGKeyArray,
    args: PyTree = None,
    sampler: eqx.Module | None = None,
    options: dict[str, Any] | None = None,
    *,
    y0: Y | None = None,  
    y_live: Y | None = None, 
    log_prior_fn: Callable | None = None,        
    prior_transform_fn: Callable | None = None,  
    prior_sample_fn: Callable | None = None,
    nlive: int | None = None,
    max_steps: int | None = None,
    batch_size: int = 2000,
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
        return sampler(log_likelihood_fn, prior_transform_fn, y0, args, nlive=nlive)

    elif isinstance(sampler, AbstractHostPhysicalNS):
        if log_prior_fn is None:
            raise ValueError(f"{sampler.__class__.__name__} requires `log_prior_fn`.")
        return sampler(log_likelihood_fn, log_prior_fn, y0, args, nlive=nlive)

    # --- 2. INITIALIZE LIVE POINTS ---
    if y_live is None:
        if y0 is None or nlive is None:
            raise ValueError("Must provide either `y_live` OR both `y0` and `nlive`.")
        
        init_key, key = jax.random.split(key)
        
        flat_y0, reconstruct_fn = jax.flatten_util.ravel_pytree(y0)
        ndims = flat_y0.size
        u_live_flat = jax.random.uniform(init_key, shape=(nlive, ndims))
        
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

    # --- 4. EXECUTE BATCHED LOOP ---
    final_buffer, final_state, final_steps, converged, status = _batched_loop(
        likelihood_func, prior_func, sampler, y_live, args, key, options, max_steps, batch_size
    )

    # --- 5. POST-PROCESSING ---
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