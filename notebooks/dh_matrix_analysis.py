
from numba import njit, prange
from linear_system_super_resolution import *

### ===================================================================================================================
### Full approach
### ===================================================================================================================

def generate_dh_matrix(kernel, high_res_dim, downsample):
    
    low_res_dim = high_res_dim // downsample
    
    # %lprun -f calculate_d_origins d_origins = calculate_d_origins(high_res_dim, downsample)
    d_origins = calculate_d_origins(high_res_dim, downsample)
    
    # %lprun -f produce_stacked_kernel stacked_kernel = produce_stacked_kernel(kernel, downsample)
    stacked_kernel = produce_stacked_kernel(kernel, downsample)
    # show_image(stacked_kernel, "Stacked")
    
    # Fill in entries for sparse DH matrix
    # %lprun -f populate_dh_buffers row_buffer, col_buffer, val_buffer = populate_dh_buffers(d_origins, stacked_kernel, high_res_dim // downsample, high_res_dim, kernel.shape[0])
    row_buffer, col_buffer, val_buffer = populate_dh_buffers(d_origins, stacked_kernel, low_res_dim, high_res_dim, kernel.shape[0])
    
    return coo_matrix((val_buffer, (row_buffer, col_buffer)), shape=(low_res_dim**2, high_res_dim**2), dtype=np.float32)

# @njit(parallel = True)
def populate_dh_buffers(d_origins, kernel, low_res_dim, high_res_dim, original_kernel_samples):
    
    # To do: reevaluate how the dh matrix is populated...
    # ... ideally now instead of just clipping some inner region of the flattened kernel (using left/right clips)
    # ... I will now need to consider ignoring values within the inner region which would technically fall out of bounds for traditional unpadded convolution
    # ... it still makes sense to do a double sweep I think but now how I determine total_samples will differ as well as the populating process
    # ... Maybe I can do some process where I take the pixel location (i) for low_res_dim and 'map' it to a square of ints around that location but mapped to pixels of high_res_dim
    
    kernel_samples = kernel.shape[0]
    kernel_offset = ((original_kernel_samples - 1) // 2) * (high_res_dim + 1)
    kernel = kernel.flatten()
    
    buffer_strides = np.zeros(low_res_dim**2, dtype=np.uintc)
    # left_clip = np.zeros(low_res_dim**2, dtype=np.uintc)
    # right_clip = np.zeros(low_res_dim**2, dtype=np.uintc)
    samples_per_dh_row = np.zeros(low_res_dim**2, dtype=np.uintc)
    
    # First pass to determine how big to make row/col/val buffers and the stride used for each thread to populate each respective row of DH
    for i in prange(low_res_dim**2):
        repeated_range = np.repeat(np.arange(kernel_samples, dtype=np.intc), kernel_samples)
        cols = repeated_range.reshape(-1, kernel_samples).T.flatten() + (repeated_range * high_res_dim) + d_origins[i] - kernel_offset
        exclusion_mask = matrix_edge_exclusion_mask(d_origins[i], high_res_dim, kernel_samples, original_kernel_samples)
        # print(f"Cols: {cols.reshape(kernel_samples, kernel_samples)}")
        # print(f"Exclusion mask: {exclusion_mask}")
        cols = cols[exclusion_mask == 1]
        # print(f"Mask applied: {cols}")
        # print("\n\n")
        # left_clip[i] = cols[cols < 0].shape[0]
        # right_clip[i] = cols[cols >= high_res_dim**2].shape[0]
        samples_per_dh_row[i] = cols.shape[0]
    
    total_samples = samples_per_dh_row.sum()
    # samples_per_clipped_dh_row = kernel_samples**2 - (left_clip + right_clip)
    # total_samples = samples_per_clipped_dh_row.sum()
    
    # Below for Numba annotated func, cumsum doesnt support type...
    # buffer_strides = np.append([0], np.cumsum(samples_per_clipped_dh_row))
    # Below for regular func, cumsum requires dtype...
    # buffer_strides = np.append([0], np.cumsum(samples_per_clipped_dh_row, dtype=np.uintc)) # prepend 0 to allow for ranges
    buffer_strides = np.append([0], np.cumsum(samples_per_dh_row, dtype=np.uintc)) # prepend 0 to allow for ranges
    row_buffer = np.zeros(total_samples, dtype=np.uintc)
    col_buffer = np.zeros(total_samples, dtype=np.uintc)
    val_buffer = np.zeros(total_samples, dtype=np.float32)
    
    # Second pass to populate the row/col/val buffers using predetermined strides and clipping parameters
    for i in prange(low_res_dim**2):
        repeated_range = np.repeat(np.arange(kernel_samples, dtype=np.intc), kernel_samples)
        cols = repeated_range.reshape(-1, kernel_samples).T.flatten() + (repeated_range * high_res_dim) + d_origins[i] - kernel_offset
        exclusion_mask = matrix_edge_exclusion_mask(d_origins[i], high_res_dim, kernel_samples, original_kernel_samples)
        cols = cols[exclusion_mask == 1]
        row_buffer[buffer_strides[i] : buffer_strides[i+1]] = i
        col_buffer[buffer_strides[i] : buffer_strides[i+1]] = cols
        val_buffer[buffer_strides[i] : buffer_strides[i+1]] = kernel[exclusion_mask == 1]
        
    return row_buffer, col_buffer, val_buffer

### ===================================================================================================================
### 
### ===================================================================================================================




### ===================================================================================================================
### Batched approach
### ===================================================================================================================

def generate_dh_matrix_batched(kernel, high_res_dim, downsample, dh_matrix_batch_size):
    
    low_res_dim = high_res_dim // downsample
    
    # %lprun -f calculate_d_origins d_origins = calculate_d_origins(high_res_dim, downsample)
    d_origins = calculate_d_origins(high_res_dim, downsample)
    
    # %lprun -f produce_stacked_kernel stacked_kernel = produce_stacked_kernel(kernel, downsample)
    stacked_kernel = produce_stacked_kernel(kernel, downsample)
    # show_image(stacked_kernel, "Stacked")
    
    # Placeholder matrix to be populated over time
    m = coo_matrix((low_res_dim**2, high_res_dim**2), dtype=np.float32)

    # Fill in entries for sparse DH matrix
    for b in range(low_res_dim**2 // dh_matrix_batch_size):
        # %lprun -f populate_dh_buffers row_buffer, col_buffer, val_buffer = populate_dh_buffers(d_origins, stacked_kernel, high_res_dim // downsample, high_res_dim, kernel.shape[0])
        row_buffer, col_buffer, val_buffer = populate_dh_buffers_batched(d_origins, stacked_kernel, low_res_dim, high_res_dim, kernel.shape[0], dh_matrix_batch_size, b * dh_matrix_batch_size)
        m += coo_matrix((val_buffer, (row_buffer, col_buffer)), shape=(low_res_dim**2, high_res_dim**2), dtype=np.float32)
        # show_image(m.todense(), f"Matrix (Batch {b})")
    
    return m

# @njit(parallel = True)
def populate_dh_buffers_batched(d_origins, kernel, low_res_dim, high_res_dim, original_kernel_samples, batch_size, batch_offset):
    
    kernel_samples = kernel.shape[0]
    kernel_offset = ((original_kernel_samples - 1) // 2) * (high_res_dim + 1)
    kernel = kernel.flatten()
    
    buffer_strides = np.zeros(batch_size, dtype=np.uintc)
    samples_per_dh_row = np.zeros(low_res_dim**2, dtype=np.uintc)
    
    # First pass to determine how big to make row/col/val buffers and the stride used for each thread to populate each respective row of DH
    for i in prange(batch_size):
        repeated_range = np.repeat(np.arange(kernel_samples).astype(np.intc), kernel_samples)
        cols = repeated_range.reshape(-1, kernel_samples).T.flatten() + (repeated_range * high_res_dim) + d_origins[i+batch_offset] - kernel_offset
        exclusion_mask = matrix_edge_exclusion_mask(d_origins[i+batch_offset], high_res_dim, kernel_samples, original_kernel_samples)
        cols = cols[exclusion_mask == 1]
        samples_per_dh_row[i] = cols.shape[0]
    
    total_samples = samples_per_dh_row.sum()
    
    # Below for Numba annotated func, cumsum doesnt support type...
    # buffer_strides = np.append([0], np.cumsum(samples_per_dh_row))
    # Below for regular func, cumsum requires dtype...
    buffer_strides = np.append([0], np.cumsum(samples_per_dh_row, dtype=np.uintc)) # prepend 0 to allow for ranges
    
    row_buffer = np.zeros(total_samples, dtype=np.uintc)
    col_buffer = np.zeros(total_samples, dtype=np.uintc)
    val_buffer = np.zeros(total_samples, dtype=np.float32)
    
    # Second pass to populate the row/col/val buffers using predetermined strides and clipping parameters
    for i in prange(batch_size):
        repeated_range = np.repeat(np.arange(kernel_samples).astype(np.intc), kernel_samples)
        cols = repeated_range.reshape(-1, kernel_samples).T.flatten() + (repeated_range * high_res_dim) + d_origins[i+batch_offset] - kernel_offset
        exclusion_mask = matrix_edge_exclusion_mask(d_origins[i+batch_offset], high_res_dim, kernel_samples, original_kernel_samples)
        cols = cols[exclusion_mask == 1]
        row_buffer[buffer_strides[i] : buffer_strides[i+1]] = i+batch_offset
        col_buffer[buffer_strides[i] : buffer_strides[i+1]] = cols
        val_buffer[buffer_strides[i] : buffer_strides[i+1]] = kernel[exclusion_mask == 1]
        
    return row_buffer, col_buffer, val_buffer

### ===================================================================================================================
### 
### ===================================================================================================================

def produce_stacked_kernel(kernel, downsample):
    kernel_dim = kernel.shape[0]
    stacked = np.zeros((kernel_dim + downsample - 1, kernel_dim + downsample - 1), dtype=np.float32)
    for r in range(downsample):
        for c in range(downsample):
            stacked[r:r+kernel_dim, c:c+kernel_dim] += kernel
    
    return stacked

# Calculates the first column index per row of D if we were to use D as a genuine downsampling matrix
def calculate_d_origins(high_res_dim, downsample):
    low_res_dim = high_res_dim // downsample
    return np.tile(np.arange(0, high_res_dim, downsample, dtype=np.uintc), low_res_dim) + np.repeat(np.arange(low_res_dim, dtype=np.uintc) * high_res_dim * downsample, low_res_dim)

def matrix_edge_exclusion_mask(current_index, high_res_dim, stacked_kernel_dim, original_kernel_dim):
    current_row = current_index // high_res_dim
    current_col = current_index % high_res_dim
    mask = np.ones((stacked_kernel_dim, stacked_kernel_dim))
    row_neighbours = np.arange(current_row, current_row + stacked_kernel_dim) - ((original_kernel_dim-1)//2)
    col_neighbours = np.arange(current_col, current_col + stacked_kernel_dim) - ((original_kernel_dim-1)//2)
    mask[(row_neighbours < 0) | (row_neighbours >= high_res_dim), :] = 0
    mask[:, (col_neighbours < 0) | (col_neighbours >= high_res_dim)] = 0
    
    return mask.flatten()
