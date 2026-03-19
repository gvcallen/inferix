# Inferix: A unified interface for probabilistic inference in JAX + Equinox

| **Inferix** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/gvcallen/inferix](https://github.com/gvcallen/inferix) |

## Installation
Inferix can be installed using pip directly:

``
pip install inferix
``

## Motivation

In the JAX ecosystem, you typically have to choose between two extremes for Bayesian inference:
- Wrappers around lower-level drivers (like BlackJAX or PolyChord), which force you to manually manage while-loops, PRNG keys, buffers, and algorithmic states.
- High-level Probabilistic Programming Languages (PPLs) (like NumPyro or PyMC), which are user-friendly but force you to rewrite your models using their specific domain-specific languages and distribution primitives.

The goal of `Inferix` is to be a middle option that mirrors the API of [Optimistix](https://github.com/patrick-kidger/optimistix). It is designed for engineers and scientists who already have a forward model written in pure JAX, and just want to sample from it without managing boilerplate or adopting a heavy PPL framework.

Inferix wraps low-level algorithms in a unified interface (`inferix.mcmc_sample` or `inferix.nested_sample`) and handles any host-bridge, XLA-compiled control flow, reparameterizations and data packaging. Current kernels include JAX-native NUTS and Nested Slice Sampling (via `BlackJAX`), and a host-bridged `PolyChord`.

```python
import jax
import jax.numpy as jnp
import inferix

# 1. Define your target functions (Pure JAX)
def my_likelihood(theta, args):
    # e.g., A complex differentiable physics simulation
    return ... 

def my_prior_transform(u, args):
    # A mapping from the uniform unit hypercube coordinates u to physical parameters theta
    return ...

# 2. Instantiate your sampler of choice e.g. inferix.NUTS, inferix.NSS or inferix.PolyChord
sampler = inferix.NSS(num_delete=10, num_inner_steps=20)

# 3. Execute the run
key = jax.random.PRNGKey(42)
result = inferix.nested_sample(
    log_likelihood_fn=my_likelihood,
    prior_transform_fn=my_prior_transform,
    sampler=sampler,
    ndims=5,
    key=key,
    logZ_convergence=1e-3,
)

# 4. Access the results
print(f"Final log-Evidence (logZ): {result.logZ} ± {result.logZ_err}")
print(f"Num samples: {result.samples.shape[0]}")
```