
import numpy as np
from scipy.sparse import bsr_matrix

def convolution_matrix_sparse(l, kernel):
    
    # Allocate a key-value pair for each distinct kernel sample (key) to an empty list (value) 
    pixel_to_row_col_dict = {val : [] for val in np.unique(kernel)}
    full_supp = kernel.shape[0] # assumed square
    half_supp = (full_supp - 1) // 2
    
    kernel = kernel.flatten()
    neighbour_strides = np.arange(-(half_supp), half_supp+1)
    
    for m_row in np.arange(l**2):
        
        # map flattened m_row to 2d row/col  
        row, col = (m_row // l, m_row % l)
        # print(f"M_row {m_row} maps to row/col {row}, {col}")
        
        neighbour_rows = neighbour_strides + row
        neighbour_cols = neighbour_strides + col
        
        # map kernel to neighbouring indices of row/col, from centre of kernel
        meshgrid = np.meshgrid(neighbour_rows, neighbour_cols, copy=False)
        mesh = np.array(meshgrid)
        neighbour_indices = mesh.T.reshape(-1, 2)
        
        # zero out in kernel, any neighbours which have negative index or greater than l
        bad_rows = np.concatenate((np.where(neighbour_indices[:, 0] < 0)[0], np.where(neighbour_indices[:, 0] >= l)[0]))
        bad_cols = np.concatenate((np.where(neighbour_indices[:, 1] < 0)[0], np.where(neighbour_indices[:, 1] >= l)[0]))
        # get a set of distinct neighbours to be zeroed out (in case of duplicate entries where a bad row is also a bad column)
        bad_neighbours = np.unique(np.concatenate((bad_rows, bad_cols)))
        # print(f"Bad neighbours: {bad_neighbours}")

        modified_kernel = np.copy(kernel)
        # print(f"Original kernel: {modified_kernel}")
        modified_kernel = np.delete(modified_kernel, bad_neighbours)
        # print(f"Modified kernel: {modified_kernel}")
        col_strides = np.repeat(neighbour_strides, (np.repeat(full_supp, full_supp))) * (l - full_supp)
        # print(f"Col Strides: {col_strides}")
        cols = np.delete((np.arange(m_row, m_row + full_supp**2) - full_supp**2 // 2 + col_strides), bad_neighbours)
        
        # print(f"Column vals: {cols}")
        for index, entry in enumerate(modified_kernel):
            pixel_to_row_col_dict[entry].append((m_row, cols[index]))
        
    entries = sum(len(v) for v in pixel_to_row_col_dict.values())
    # print(entries)

    row_col_pairs = np.zeros((entries, 2), dtype=np.uintc)
    kernel_samples = np.zeros(entries, dtype=np.float32)
    stride = 0
    for k, v in pixel_to_row_col_dict.items():

        if not v: # ignore pixels with no entries
            continue

        multiplier = len(v)
        # print(multiplier)
        kernel_samples[stride : stride + multiplier] = k
        row_col_pairs[stride : stride + multiplier] = v
        stride += multiplier
    
    return bsr_matrix((kernel_samples, (row_col_pairs[:, 0], row_col_pairs[:, 1])), shape=(l**2, l**2))
