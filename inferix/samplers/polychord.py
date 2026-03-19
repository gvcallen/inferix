from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from inferix.nested import AbstractHostHypercubeNS, Result

class PolyChord(AbstractHostHypercubeNS):
    """
    PolyChord Nested Sampling wrapper.
    
    Exposes all native PolyChord settings. Attributes set to `None` 
    will automatically fall back to PolyChord's dimension-dependent defaults
    (e.g., nlive defaults to ndims * 25).
    """
    
    # --- Dynamic / Dimension-Dependent Defaults ---
    nlive: Optional[int] = None
    num_repeats: Optional[int] = None
    grade_dims: Optional[List[int]] = None
    grade_frac: Optional[List[float]] = None
    nlives: Optional[Dict[float, int]] = None
    cube_samples: Optional[Any] = None  # Expected array-like
    
    # --- Static Defaults ---
    nprior: int = -1
    nfail: int = -1
    do_clustering: bool = True
    feedback: int = 1
    precision_criterion: float = 0.001
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
    seed: int = -1

    def run(
        self, 
        log_likelihood_fn: Callable, 
        prior_transform_fn: Callable, 
        ndims: int, 
        args: PyTree
    ) -> Result: # type: ignore
        
        try:
            import pypolychord
            from pypolychord.settings import PolyChordSettings
        except ImportError:
            raise ImportError("pypolychord must be installed to use this sampler.")

        # 1. Resolve Dynamic Defaults
        _nlive = self.nlive if self.nlive is not None else ndims * 25
        _num_repeats = self.num_repeats if self.num_repeats is not None else ndims * 5
        _grade_dims = self.grade_dims if self.grade_dims is not None else [ndims]
        _grade_frac = self.grade_frac if self.grade_frac is not None else [1.0] * len(_grade_dims)
        _nlives = self.nlives if self.nlives is not None else {}

        # 2. Populate PolyChordSettings
        settings = PolyChordSettings(nDims=ndims, nDerived=0)
        settings.nlive = _nlive
        settings.num_repeats = _num_repeats
        settings.nprior = self.nprior
        settings.nfail = self.nfail
        settings.do_clustering = self.do_clustering
        settings.feedback = self.feedback
        settings.precision_criterion = self.precision_criterion
        settings.logzero = self.logzero
        settings.max_ndead = self.max_ndead
        settings.boost_posterior = self.boost_posterior
        settings.posteriors = self.posteriors
        settings.equals = self.equals
        settings.cluster_posteriors = self.cluster_posteriors
        settings.write_resume = self.write_resume
        settings.write_paramnames = self.write_paramnames
        settings.read_resume = self.read_resume
        settings.write_stats = self.write_stats
        settings.write_live = self.write_live
        settings.write_dead = self.write_dead
        settings.write_prior = self.write_prior
        settings.maximise = self.maximise
        settings.compression_factor = self.compression_factor
        settings.synchronous = self.synchronous
        settings.base_dir = self.base_dir
        settings.file_root = self.file_root
        settings.seed = self.seed
        settings.grade_dims = _grade_dims
        settings.grade_frac = _grade_frac
        settings.nlives = _nlives
        settings.cube_samples = self.cube_samples

        # 3. JIT Compile the Target Functions
        jitted_likelihood = jax.jit(lambda theta: log_likelihood_fn(theta, args))
        jitted_prior = jax.jit(lambda u: prior_transform_fn(u, args))

        # 4. Create Host-to-Device Callbacks
        def polychord_likelihood(theta_np):
            logL = jitted_likelihood(jnp.asarray(theta_np))
            return float(logL), []

        def polychord_prior(u_np):
            return np.array(jitted_prior(jnp.asarray(u_np)))

        # 5. Execute Run
        output = pypolychord.run_polychord(
            loglikelihood=polychord_likelihood,
            nDims=ndims,
            nDerived=0,
            settings=settings,
            prior=polychord_prior
        )

        # 6. Package and Return
        return Result(
            samples=jnp.array(output.samples), 
            logZ=jnp.array(output.logZ),
            logZ_err=jnp.array(output.logZerr),
            num_steps=len(output.samples),
            converged=jnp.array(True),
            final_state=None
        )