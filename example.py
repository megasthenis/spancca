import numpy
import spancca

# Generate synthetic input.
dim1 = 10**4
dim2 = 1000
A = numpy.random.rand(dim1, dim2)


# The rank of the approximation to be used:
rank = 3

# The number T of samples to consider (or the number of 'iterations' of our
# algorithm). More samples means more time, but potentially better results.
T = 10**4

# Set up the projection method for the left component:
u_project = spancca.projections.setup_sparse(nnz=int(dim1/10))
# OTHER OPTIONS:
# u_project = spancca.projections.setup_l2unit()
# u_project = spancca.projections.setup_nonnegative()
u_project = spancca.projections.setup_nonnegative_sparse(nnz=int(dim1/20))

# Set up the projection method for the right component:
v_project = spancca.projections.setup_l2unit()
# OTHER OPTIONS: Similar to u.

# RUN
u, v = spancca.cca(A, rank, T, u_project, v_project, verbose=True)
