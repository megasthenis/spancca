from __future__ import print_function
from contextlib import contextmanager
import numpy
import scipy.sparse.linalg
import time
try:
    # Module for progress bar (https://github.com/mitsuhiko/click)
    import click
    _click_loaded = True
except ImportError:
    _click_loaded = False


@contextmanager
def _simplecontextgenerator(gen):
    """Simple context manager wrapper for generator.

    Args:
        gen: generator or iterator.
    """
    try:
        yield gen
    finally:
        pass


def _normalize(v):
    """Scale a vector to have unit l2 norm."""

    norm = numpy.linalg.norm(v, ord=2)
    if norm > 0:
        return v / norm

    return v


def _candidate(U, V, c, u_project, v_project):
    """Generate a pair of components (rank-1 instance)."""

    # Generate sample point in range{U}.
    a = numpy.dot(U, c)
    # Project point on the constraint set of u to obtain the left component.
    u = u_project(a)
    u = _normalize(u)

    # "Alternate" to obtain right component v:
    b = numpy.dot(U.transpose(), u)
    b = numpy.dot(V, b)
    v = v_project(b)
    v = _normalize(v)

    # Objective value
    obj = numpy.dot(b.transpose(), v)
    return u, v, obj


def cca(A, rank, T, u_project, v_project, verbose=True):
    """Compute structered canonical correlation components.

    Args:
        A (numpy.array): The input (cross-correlation) matrix.
        approx_rank (int): the rank of the approximation of A to be used in
            the bilinear maximization problem.
        T (int): the number of subspace samples to be considered in the
            bilinear maximization.
        u_project: the projection operator to be used for the left component.
            Projection method can be selected from spancca.projections.
        v_project: the projection operator to be used for the right component.
            Projection method can be selected from spancca.projections.

    Returns:
        (numpy.array, numpy.array): the left and right components.
    """

    t_init = time.time()

    # Compute the principal subspace of the input matrix A.
    # Apply truncated singular value decomposition:
    if verbose:
        print('Computing SVD...', end=""),

    # scipy.sparse.linalg.svds supports output with number of components k at
    # most equal to rank(A)-1.
    U, S, Vt = scipy.sparse.linalg.svds(
        A,  # input matrix.
        k=rank,  # number of components.
        ncv=None, tol=0,
        which='LM',  # largest magnitude singular values.
        v0=None, maxiter=None,
        return_singular_vectors=True)

    # Reform the output of the factorization:
    # - Reorder singular values in decreasing order.
    # - Reorder singular vectors accordingly.
    # - Absorb singular values into left singular vectors (U).
    order = numpy.argsort(S)[::-1]
    S = S[order]
    U = numpy.dot(U[:, order], numpy.diag(S))
    V = (Vt[order]).transpose()

    if verbose:
        print('[Done, (%f seconds)]' % (time.time() - t_init))

    cur_obj = -float('Inf')
    cur_u = None
    cur_v = None

    with (click.progressbar(range(T), label='Sampling')
          if _click_loaded and verbose
          else _simplecontextgenerator(range(T))) as iterator:

        for i in iterator:

            # Generate random vector c on unit sphere:
            c = _normalize(numpy.random.rand(rank, 1))

            # Generate candidate solution pair:
            u, v, obj = _candidate(U, V, c, u_project, v_project)

            # Update best solution:
            if obj > cur_obj:
                cur_obj, cur_u, cur_v = obj, u, v

    if verbose:
        # Output info
        def vec_info(x):
            s = 'dim=%s, |supp|=%d, l2=%f' % (
                x.shape, numpy.count_nonzero(x), numpy.linalg.norm(x, ord=2))
            return s

        print('[Summary]')
        print('Input: \t', end="")
        print('A: %d x %d' % (A.shape[0], A.shape[1]))
        print('\tApproximation rank (r): %d' % rank)
        print('\tSing. values: s(1)=%f, s(2)=%f, ..., s(r)=%f' % (
            tuple(S[[0, 1, rank-1]].flatten())))
        print('Output:\t', end="")
        print('Left (u): %s' % vec_info(u))
        print('\tRight (v): %s' % vec_info(v))
        print('\tObjective (u\'Av): %f' % (u.transpose().dot(A)).dot(v))
        print('[Total time: %d seconds.]' % (time.time() - t_init))

    return cur_u, cur_v
