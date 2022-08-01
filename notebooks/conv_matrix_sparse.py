
from memory_profiler import profile
import numpy as np
from scipy.sparse import bsr_matrix, diags, dia_matrix, csr_matrix

# Steps to map a 2d convolution kernel to a "convolution" matrix. 
# Note: this disregards padding of the convolution at edges of the matrix to be "convolved" with. This is essentially a diagonal matrix, where the 2d kernel
#       is flattened, and each 1d "slice" is strided from one another by a separation of around "convolution" matrix width.
# Note: this method assumes we can build up a subset of the full matrix (lets call it a "submatrix"), and use simply arithmetic to
#       "slide" the submatrix diagonally to populate the full "convolution" matrix.
# Note: The format for storing this matrix is the Block Compressed Row sparse matrix format:
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html

@profile
def conv_matrix_sparse(l, kernel):
    supp = kernel.shape[0]
    half_supp = (supp - 1) // 2
    diagonal_strides = np.linspace(-half_supp, half_supp, supp, dtype=np.intc)

    submatrix_cols = np.zeros([], dtype=np.uintc)
    submatrix_rows = np.zeros([], dtype=np.uintc)
    submatrix_elements = np.zeros([], dtype=np.float32)

    for k in range(supp):
        # generate unpadded diagonal sparse matrix of kth kernel slice
        k_bsr = dia_matrix.tocsr(diags(kernel[k], diagonal_strides, shape=(l, l)))

        cols_for_rows = np.split(k_bsr.indices, k_bsr.indptr[1:-1]) # get col indices for entries in matrix
        rows_for_cols = np.repeat(np.arange(l), [len(arr) for arr in cols_for_rows])
        cols_for_rows = np.concatenate(cols_for_rows).ravel() + k * l # offset each diagonal

        # Ideally use of pre-allocated memory of known dimensions is ideal... but okay for now
        submatrix_cols = np.append(submatrix_cols, cols_for_rows)
        submatrix_rows = np.append(submatrix_rows, rows_for_cols)
        submatrix_elements = np.append(submatrix_elements, k_bsr.data)

    matrix_cols = np.zeros([], dtype=np.uintc)
    matrix_rows = np.zeros([], dtype=np.uintc)
    matrix_elements = np.zeros([], dtype=np.float32)

    # Now that we have the initial "submatrix" of diagonals, we can populate
    # the sparse matrix using strided, truncated submatrices
    for index, stride in enumerate(np.arange(-half_supp, -half_supp + l) * l):
        strided_cols = submatrix_cols + stride
        in_range_cols = np.where((strided_cols >= 0) & (strided_cols < l**2))
        strided_rows = submatrix_rows + index * l
        valid_elements = submatrix_elements

        # This needs changing at some point, highly inefficient...
        matrix_cols = np.append(matrix_cols, strided_cols[in_range_cols])
        matrix_rows = np.append(matrix_rows, strided_rows[in_range_cols])
        matrix_elements = np.append(matrix_elements, submatrix_elements[in_range_cols])

    return csr_matrix((matrix_elements, (matrix_rows, matrix_cols)), shape=(l**2, l**2))
