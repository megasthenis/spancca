import numpy
import unittest
import projections as ps


class TestProjectionMethods(unittest.TestCase):

    def test_project_l2_unit_works(self):

        v = numpy.array([2.0, 1.0, 4.0])
        actual = ps._project_l2_unit(v)
        expect = numpy.array([2.0, 1.0, 4.0]) / numpy.sqrt(21)
        numpy.testing.assert_array_equal(actual, expect)

        # Test vector that lies inside the unit ball:
        v = numpy.array([0.0, 0.1, 0.2, 0.4])
        actual = ps._project_l2_unit(v)
        expect = v
        numpy.testing.assert_array_equal(actual, expect)

        # Test all-zero vector.
        v = numpy.array([0.0, 0.0, 0.0, 0.0])
        actual = ps._project_l2_unit(v)
        expect = v
        numpy.testing.assert_array_equal(actual, expect)

    def test_project_sparse_works(self):

        nnz = 2
        # Check on flattened ndarray
        v = numpy.array([1., 5., 3., 4., 2.])
        actual = ps._project_sparse(v, nnz)
        expect = numpy.array([0., 5., 0., 4., 0.])
        numpy.testing.assert_array_equal(actual, expect)

        # Check on row vector.
        v = numpy.array([1, -5, 3, 4, 2], ndmin=2)
        actual = ps._project_sparse(v, nnz)
        expect = numpy.array([0, -5, 0, 4, 0], ndmin=2)
        numpy.testing.assert_array_equal(actual, expect)

        # Check on column vector and normalize:
        v = numpy.array([1., -5., 3., 4., 2.], ndmin=2).transpose()
        actual = ps._project_sparse(v, nnz, normalize=True)
        expect = numpy.array([0., -5., 0., 4., 0.]) / numpy.sqrt(41)
        expect.shape = (5, 1)
        numpy.testing.assert_array_equal(actual, expect)

        # Check on all zeros vector.
        v = numpy.array([0.0, 0.0, 0.0, 0.0], ndmin=2)
        actual = ps._project_sparse(v, nnz, normalize=True)
        expect = numpy.array([0.0, 0.0, 0.0, 0.0], ndmin=2)
        numpy.testing.assert_array_equal(actual, expect)

    def test_project_nonnegative_sparse_works(self):

        # Check nonnegative input, no sparsity (flattened ndarray):
        v = numpy.array([1.0, 5.0, 3.0, 4.0, 2.0])
        actual = ps._project_nonnegative_sparse(v, None)
        expect = numpy.array([1.0, 5.0, 3.0, 4.0, 2.0])
        numpy.testing.assert_array_equal(actual, expect)

        # Check nonnegative, with sparsity (use column vector input):
        nnz = 3
        v = numpy.array([1.0, 5.0, 3.0, 4.0, 2.0], ndmin=2).transpose()
        actual = ps._project_nonnegative_sparse(v, nnz)
        expect = numpy.array([0., 5.0, 3.0, 4.0, 0.0], ndmin=2).transpose()
        numpy.testing.assert_array_equal(actual, expect)

        # Check arbitrary input, no sparsity (use row vector input):
        v = numpy.array([1.0, -5.0, 3.0, 4.0, 2.0], ndmin=2)
        actual = ps._project_nonnegative_sparse(v, None)
        expect = numpy.array([1.0, 0.0, 3.0, 4.0, 2.0], ndmin=2)
        numpy.testing.assert_array_equal(actual, expect)

        # Check arbitrary input, with sparsity parameter. Target nnz is equal
        # to size of the entire input vector. Scale output to unit l2 length.
        nnz = 5
        v = numpy.array([1.0, -5.0, 3.0, 4.0, 2.0], ndmin=2)
        actual = ps._project_nonnegative_sparse(v, nnz, normalize=True)
        expect = (numpy.array([1.0, 0.0, 3.0, 4.0, 2.0], ndmin=2) /
                  numpy.sqrt(30))
        numpy.testing.assert_array_equal(actual, expect)

        # Check arbitrary input, with sparsity parameter. Target nnz is less
        # than available positive entries. Scale output to unit l2 length.
        nnz = 2
        v = numpy.array([1.0, -5.0, 3.0, 4.0, -2.0], ndmin=2)
        actual = ps._project_nonnegative_sparse(v, nnz, normalize=True)
        expect = (numpy.array([0.0, 0.0, 3.0, 4.0, 0.0], ndmin=2) /
                  numpy.sqrt(25))
        numpy.testing.assert_array_equal(actual, expect)

if __name__ == '__main__':
    unittest.main()
