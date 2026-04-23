Inferix: An (experimental) unified interface for probabilistic inference in JAX + Equinox

| **Inferix** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/gvcallen/inferix](https://github.com/gvcallen/inferix) |

NB: This library is in very early stages, and is likely not currently functional.

## Installation
Inferix can be installed using pip directly:

``
pip install inferix
``

## Motivation

In the JAX ecosystem, you typically have to choose between two extremes for Bayesian inference:
- Lower-level drivers (like BlackJAX), where you have to manually manage while-loops, PRNG keys, buffers, and algorithmic states.
- High-level Probabilistic Programming Languages (PPLs) (like NumPyro or jaxns), which are user-friendly but requiring rewriting your model using their specific domain-specific languages and distribution primitives.

The goal of `Inferix` is to be a middle option that mirrors the API of [Optimistix](https://github.com/patrick-kidger/optimistix). It is designed for engineers and scientists who already have a forward likelihood and prior model written in pure JAX (perhaps using [Equinox](https://github.com/patrick-kidger/equinox)), and just want to sample from it without managing boilerplate or adopting a heavy PPL framework.

Inferix wraps low-level algorithms in a unified interface (`inferix.mcmc_sample` or `inferix.nested_sample`) and handles any XLA-compiled control flow, reparameterizations and data packaging. Current kernels include JAX-native NUTS and Nested Slice Sampling (via `BlackJAX`).

```python
import jax
import jax.numpy as jnp
import inferix

# 1. Define your target functions (Pure JAX)
def my_likelihood(theta, args):
    # e.g., A physics-based model
    return ... 

def my_prior_transform(u, args):
    # A mapping from the uniform unit hypercube coordinates u to physical parameters theta
    return ...

# 2. Instantiate your sampler of choice e.g. inferix.NUTS or inferix.NSS
sampler = inferix.NSS(num_delete=10, num_inner_steps=20, logZ_convergence=1e-3)

# 3. Execute the run
key = jax.random.key(42)
result = inferix.nested_sample(
    log_likelihood_fn=my_likelihood,
    prior_transform_fn=my_prior_transform,
    sampler=sampler,
    key=key,
)

# 4. Access the results
print(f"Final log-Evidence (logZ): {result.logZ} ± {result.logZ_err}")
print(f"Num samples: {result.samples.shape[0]}")
```
