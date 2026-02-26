"""
jit_utils.py — JIT compilation utilities for Myanmar highway alignment
======================================================================
Provides a `try_jit()` decorator that:
  - If numba is installed:  wraps with @numba.njit(cache=True, fastmath=True)
    → first-call compile, then 10-50x faster than CPython for inner loops.
  - If numba is NOT installed: is a transparent no-op — function runs as
    plain Python/numpy with zero overhead and zero API change.

Usage:
    from jit_utils import try_jit

    @try_jit
    def heavy_loop(arr):
        ...

Install numba:
    pip install numba>=0.57
After install, the next pipeline run will compile and cache all decorated
functions. Subsequent runs skip compilation (cache=True).
"""
import logging

log = logging.getLogger("highway_alignment")

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
    log.info(f"numba {_numba.__version__} detected — JIT compilation active.")
except ImportError:
    _NUMBA_AVAILABLE = False
    log.debug("numba not installed — running in pure Python/numpy mode.")


def try_jit(fn=None, *, parallel=False, fastmath=True, cache=True):
    """
    Decorator factory: applies numba.njit when available, else no-op.

    Can be used as bare @try_jit or with arguments @try_jit(parallel=True).

    Parameters
    ----------
    parallel : bool
        Enable numba's automatic parallelisation (prange). Requires that the
        function uses numba.prange instead of range in parallel loops.
    fastmath : bool
        Allow reassociation of float ops (slight precision loss, ~20% faster).
    cache : bool
        Cache compiled bytecode to disk — eliminates recompile on next run.
    """
    def _decorator(func):
        if not _NUMBA_AVAILABLE:
            return func   # transparent pass-through
        try:
            jitted = _numba.njit(
                func, cache=cache, fastmath=fastmath, parallel=parallel
            )
            log.debug(f"JIT-compiled: {func.__qualname__}")
            return jitted
        except Exception as exc:
            log.warning(
                f"numba JIT failed for {func.__qualname__} ({exc}). "
                f"Using Python fallback."
            )
            return func

    # Allow both @try_jit and @try_jit() syntaxes
    if fn is not None:
        # Called as @try_jit (no parentheses)
        return _decorator(fn)
    # Called as @try_jit(...) — return the decorator
    return _decorator


def numba_available():
    """Return True if numba is importable."""
    return _NUMBA_AVAILABLE
