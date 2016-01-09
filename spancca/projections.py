import numpy


def _project_l2_unit(v, normalize=False):
    """Project on the n-dimensional unit l2 ball.

    Compute the projection of the real n-dimensional input vector v on the set
    of vectors with l2-norm at most equal to 1. If the input vector has l2-norm
    smaller than 1, the outpu coincides with the input. Otherwise, the function
    outputs the input vector scaled to have unit length.

    Args:
        v (numpy.ndarray): the vector to be projected.
        normalize (Optional[bool]): If true, scale the output to have unit l2
            norm. Ignored if the output is the all-zero vector. Defaults to
            False.
    Returns:
        numpy.ndarray of the same shape as the input v.
    """

    # Create a replica of the input:
    v = numpy.copy(v)

    # Compute the l2 norm of the input vector:
    l2_norm = numpy.linalg.norm(v, ord=2)

    # If the input vector lies inside the l2-unit ball, do nothing:
    if l2_norm <= 1:
        return v

    # Otherwise, scale vector to unit norm:
    return v / l2_norm


def _project_sparse(v, nnz, normalize=False):
    """Project on the set of sparse vectors.

    Compute the projection of the real n-dimensional input vector v on the set
    of vectors with at most nnz nonzero entries.

    Args:
        v (numpy.ndarray): the vector to be projected.
        nnz (int): the target (upper bound on the) number of nonzero entries of
            the output vector.
        normalize (Optional[bool]): If true, scale the output to have unit l2
            norm. Ignored if the output is the all-zero vector. Defaults to
            False.
    Returns:
        numpy.ndarray of the same shape as the input v.
    """

    # Initialize the output vector p (all zeros). p is initially a flattened
    # array for convenience (It allows handling of input in the form of a
    # flattened array, or a row or column vector):
    p = numpy.zeros(v.size,)

    # Flatten input array:
    b = v.flatten()

    # Find the indices of the nnz largest (abs) entries of the input:
    ind = numpy.argpartition(numpy.absolute(b), -nnz, axis=None)[-nnz:]

    # The output vector coincides witht the input on those nnz entries:
    p[ind] = b[ind]

    if normalize is True:
        l2_norm = numpy.linalg.norm(b[ind], ord=2)
        if l2_norm > 0:
            p = p / l2_norm
        else:
            # If p is the all-zero vector, skip normalization.
            pass

    # Reshape output according to input:
    p.shape = v.shape

    return p


def _project_nonnegative_sparse(v, nnz, normalize=False):
    """Project on the set of nonnegative, sparse vectors.

    Compute the projection of the real n-dimensional input vector v on the set
    of vectors with at most nnz nonzero entries in the nonnegative orthant.

    Args:
        v (numpy.ndarray): the vector to be projected.
        nnz (int): the target (upper bound on the) number of nonzero entries of
            the output vector.
        normalize (Optional[bool]): If true, scale the output to have unit l2
            norm. Ignored if the output is the all-zero vector. Defaults to
            False.
    Returns:
        numpy.ndarray of the same shape as the input v.
    """

    if nnz is None:
        nnz = v.size

    # Flatten the input ndarray to handle all cases in the same way (e.g.,
    # input a could be a flattened array, column or row 2-d vector.):
    b = v.flatten()

    # Initialize the output vector (all zeros):
    p = numpy.zeros(v.size,)

    # Determine the subvector of positive entries:
    b_pos = (+b).clip(0)

    # Determine the indices of the nnz largest entries:
    ind_pos = numpy.argpartition(b_pos, -nnz, axis=None)[-nnz:]

    # Determine nonzero entries of the output:
    p[ind_pos] = b_pos[ind_pos]

    if normalize is True:
        norm_pos = numpy.linalg.norm(b_pos[ind_pos], ord=2)
        if norm_pos > 0:
            p = p / norm_pos
        else:
            # If p is the all-zero vector, skip normalization:
            pass

    # Reshape the output according to the input:
    p.shape = v.shape

    return p


def setup_l2unit():
    """Set up projection method on unit l2 ball."""

    def project(v):
        return _project_l2_unit(v)

    return project


def setup_sparse(nnz):
    """Set up sparse projection."""

    if nnz < 0:
        raise ValueError("nnz must be nonnegative.")

    def project(v):
        return _project_sparse(v, nnz)

    return project


def setup_nonnegative():
    """Set up nonnegative projection."""

    def project(v):
        return _project_nonnegative_sparse(v, None)

    return project


def setup_nonnegative_sparse(nnz):
    """Set up nonnegative sparse projection."""

    if nnz < 0:
        raise ValueError("nnz must be nonnegative.")

    def project(v):
        return _project_nonnegative_sparse(v, nnz)

    return project
