from typing import Any

import torch
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from einx import rearrange
from jaxtyping import Float

from .FCLSU import FCLSU
from .VCA import vca, vca_torch, vca_torch_batch

__all__ = [
    "vca_fclsu_nnls_solver",
    "cache_solver_result",
]


def vca_fclsu_nnls_solver(
    hyper_img: Float[torch.Tensor, "bands h w"],
    n_endmembers: int,
    vca_backend: str = "numpy",
    vca_batch_size: int = 4000,
    fclsu_solver_kwargs: dict = {},
):
    h, w = hyper_img.shape[-2:]
    img_1d_vca = rearrange("bands h w -> bands (h w)", hyper_img)
    img_1d_fclsu = rearrange("bands h w -> (h w) bands", hyper_img)
    if vca_backend == "torch":
        endmembers, *_ = vca_torch(
            img_1d_vca, n_endmembers, verbose=False, device=str(hyper_img.device)
        )  # [c, em]
    elif vca_backend == "torch_batch":
        endmembers, *_ = vca_torch_batch(
            img_1d_vca,
            n_endmembers,
            batch_size=vca_batch_size,
            device=str(hyper_img.device),
        )  # [c, em]
    else:  # 'numpy'
        endmembers, *_ = vca(img_1d_vca.cpu().numpy(), n_endmembers)
        endmembers = torch.as_tensor(endmembers, device=hyper_img.device)

    fclsu_abunds = FCLSU(endmembers, img_1d_fclsu, **fclsu_solver_kwargs)
    return endmembers, fclsu_abunds.reshape(n_endmembers, h, w)


def _make_solver_cache_key(
    hyper_img: Float[torch.Tensor, "bands h w"],
    n_endmembers: int,
    cache_name: str = "default",
    vca_backend: str = "numpy",
    vca_batch_size: int = 4000,
    fclsu_solver_kwargs: dict = {},
) -> tuple:
    """
    Generate a cache key for the solver function.

    Parameters
    ----------
    hyper_img : torch.Tensor
        Hyperspectral image tensor
    n_endmembers : int
        Number of endmembers
    cache_name : str
        Cache name identifier
    vca_backend : str
        VCA backend type
    vca_batch_size : int
        Batch size for VCA
    fclsu_solver_kwargs : dict
        FCLS solver kwargs

    Returns
    -------
    tuple
        Hashable cache key
    """
    # Convert torch tensor to hashable representation
    if isinstance(hyper_img, torch.Tensor):
        # Use shape, dtype, and device as key components
        img_key = (
            hyper_img.shape,
            hyper_img.dtype,
            str(hyper_img.device),
            hash(tuple(hyper_img.view(-1).tolist()[:100])),  # Sample for quick hash
        )
    else:
        img_key = id(hyper_img)

    # Convert dict kwargs to hashable form
    if fclsu_solver_kwargs:
        kwargs_key = tuple(sorted(fclsu_solver_kwargs.items()))
    else:
        kwargs_key = ()

    return hashkey(
        cache_name,
        n_endmembers,
        vca_backend,
        vca_batch_size,
        img_key,
        kwargs_key,
    )


def cache_solver_result(
    disable: bool = False,
    size: int = 1,  # deque size, usually the unmixing task uses only one image to train
):
    """
    Create a cached solver function using cachetools LRU cache.

    Parameters
    ----------
    disable : bool
        Whether to disable caching
    size : int
        Maximum cache size

    Returns
    -------
    callable
        Cached solver function with cache management methods
    """
    # Create LRU cache instance
    cache = LRUCache(maxsize=max(1, size))

    @cached(cache=cache, key=_make_solver_cache_key, info=True)
    def _cached_solver(
        hyper_img: Float[torch.Tensor, "bands h w"],
        n_endmembers: int,
        cache_name: str = "default",
        vca_backend: str = "numpy",
        vca_batch_size: int = 4000,
        fclsu_solver_kwargs: dict = {},
    ):
        """
        Core solver function that gets cached.
        """
        # Run the pipe
        endmembers, abunds = vca_fclsu_nnls_solver(
            hyper_img,
            n_endmembers,
            vca_backend=vca_backend,
            vca_batch_size=vca_batch_size,
            fclsu_solver_kwargs=fclsu_solver_kwargs,
        )
        result = (endmembers, abunds)
        return result

    def solver_wrapper(
        hyper_img: Float[torch.Tensor, "bands h w"],
        n_endmembers: int,
        cache_name: str = "default",
        vca_backend: str = "numpy",
        vca_batch_size: int = 4000,
        fclsu_solver_kwargs: dict = {},
    ):
        """
        Solver wrapper with cache management.
        """
        if disable:
            # If caching is disabled, call the function directly
            return vca_fclsu_nnls_solver(
                hyper_img,
                n_endmembers,
                vca_backend=vca_backend,
                vca_batch_size=vca_batch_size,
                fclsu_solver_kwargs=fclsu_solver_kwargs,
            )

        # Use the cached solver
        return _cached_solver(
            hyper_img,
            n_endmembers,
            cache_name,
            vca_backend,
            vca_batch_size,
            fclsu_solver_kwargs,
        )

    # Attach cache methods to the wrapper
    solver_wrapper.cache_clear = _cached_solver.cache_clear
    solver_wrapper.cache_info = _cached_solver.cache_info
    solver_wrapper.cache = cache

    return solver_wrapper
