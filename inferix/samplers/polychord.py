from typing import Any, Callable, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jaxtyping import PyTree

MPI_AVAIABLE = False
try:
    from mpi4py import MPI
    import pypolychord
    from anesthetic import NestedSamples
    MPI_AVAIABLE = True
except ImportError:
    pass

from inferix.nested import AbstractHostHypercubeNestedSampler
from inferix.result import Result, RESULTS

class PolyChord(AbstractHostHypercubeNestedSampler):
    """PolyChord Nested Sampling wrapper."""
    num_repeats: int | None = None
    nprior: int = -1
    nfail: int = -1
    do_clustering: bool = True
    feedback: int = 1
    precision_criterion: float = 1e-3
    logzero: float = -1e30
    max_ndead: int = -1
    boost_posterior: float = 0.0
    posteriors: bool = True
    equals: bool = True
    cluster_posteriors: bool = True
    write_resume: bool = True
    write_paramnames: bool = False
    read_resume: bool = True
    write_stats: bool = True
    write_live: bool = True
    write_dead: bool = True
    write_prior: bool = True
    maximise: bool = False
    compression_factor: float = np.exp(-1.0)
    synchronous: bool = True
    base_dir: str = "chains"
    file_root: str = "test"
    cluster_dir: str = "clusters"
    seed: int = -1
    nlives: Dict[float, int] = eqx.field(static=True, default_factory=dict)
    cube_samples: Any | None = None  
    paramnames: list[(str, str)] | None = None
    
    def __call__(self, log_likelihood_fn: Callable, prior_transform_fn: Callable, y0: PyTree, args: PyTree, **kwargs) -> Result:
        if not MPI_AVAIABLE:
            raise ImportError("pypolychord, anesthetic and mpi4py must be installed to use the inferix PolyChord sampler.")

        # 1. DERIVE GEOMETRY FROM y0
        flat_y0, reconstruct_fn = jax.flatten_util.ravel_pytree(y0)
        ndims = flat_y0.size

        # 2. JIT-COMPILED BRIDGES
        @jax.jit
        def jitted_likelihood(flat_theta_jax):
            struct_theta = reconstruct_fn(flat_theta_jax)
            return log_likelihood_fn(struct_theta, args)

        @jax.jit
        def jitted_prior(flat_u_jax):
            struct_u = reconstruct_fn(flat_u_jax)
            struct_theta = prior_transform_fn(struct_u, args)
            flat_theta, _ = jax.flatten_util.ravel_pytree(struct_theta)
            return flat_theta

        def polychord_likelihood(theta_np):
            logL = jitted_likelihood(jnp.asarray(theta_np))
            return float(logL), []

        def polychord_prior(u_np):
            return np.array(jitted_prior(jnp.asarray(u_np)))
        
        _dummy_prior = polychord_prior(0.5*np.ones(ndims))
        _dummy_logL = polychord_likelihood(_dummy_prior)
        
        # Combine kwargs
        base_kwargs = {
            'num_repeats': self.num_repeats if self.num_repeats is not None else ndims*5,
            'nprior': self.nprior,
            'nfail': self.nfail,
            'do_clustering': self.do_clustering,
            'feedback': self.feedback,
            'precision_criterion': self.precision_criterion,
            'logzero': self.logzero,
            'max_ndead': self.max_ndead,
            'boost_posterior': self.boost_posterior,
            'posteriors': self.posteriors,
            'equals': self.equals,
            'cluster_posteriors': self.cluster_posteriors,
            'write_resume': self.write_resume,
            'write_paramnames': self.write_paramnames,
            'read_resume': self.read_resume,
            'write_stats': self.write_stats,
            'write_live': self.write_live,
            'write_dead': self.write_dead,
            'write_prior': self.write_prior,
            'maximise': self.maximise,
            'compression_factor': self.compression_factor,
            'synchronous': self.synchronous,
            'base_dir': self.base_dir,
            'file_root': self.file_root,
            'cluster_dir': self.cluster_dir,
            'seed': self.seed,
            'nlives': self.nlives,
            'cube_samples': self.cube_samples,
            'paramnames': self.paramnames,
        }
        base_kwargs.update(kwargs)

        # 3. EXECUTE POLYCHORD
        
        nested_samples = pypolychord.run(
            loglikelihood=polychord_likelihood,
            nDims=ndims,
            nDerived=0,
            prior=polychord_prior,
            **base_kwargs,
        )
        
        exclude = ['logL', 'logL_birth', 'nlive', 'weight', 'logw']
        param_cols = [col[0] for col in nested_samples.columns if col[0] not in exclude]
        
        loglikes = jnp.array(np.array(nested_samples['logL']))
        samples = jnp.array(np.array(nested_samples.loc[:, param_cols]))
        weights = nested_samples.get_weights()

        # 4. RESTRUCTURE RESULTS -> Returns a Batch of Caller PyTrees!
        structured_samples = jax.vmap(reconstruct_fn)(jnp.array(samples))

        return Result(
            samples=structured_samples, 
            log_likelihoods=loglikes,
            final_state=None,
            log_evidence=None,
            log_evidence_err=None,
            converged=jnp.array(True),
            result=RESULTS.successful,
            weights=weights,
            stats={'output': nested_samples}
        )