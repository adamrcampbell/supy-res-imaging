
import cupy as cp
from cupyx.scipy.sparse.linalg import cg as cg_gpu
from cupyx.scipy.sparse.linalg import aslinearoperator as aslinearoperator_gpu
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
# from cupyx.scipy.sparse.linalg import lsqr as cuda_sparse_solve

from numba import njit, prange

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from scipy.stats import norm
from scipy.sparse import csr_matrix, bsr_matrix, dia_matrix, diags, csr_array, dok_matrix, coo_array, coo_matrix, csc_matrix
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr, gcrotmk, tfqmr, lsqr, lsmr, aslinearoperator
from scipy.linalg import solve
import scipy.stats as st
from skimage import data, io, color
from skimage.transform import resize

from joblib import Parallel, delayed

def numpy_memory(a):
    return (a.size * a.itemsize) / 10**6

def matrix_memory(m):
    return (m.data.nbytes + m.indptr.nbytes + m.indices.nbytes) / 10**6 # MB

def matrix_density(m):
    return m.getnnz() / np.prod(m.shape)

def show_image(image, title, flip_x_axis=False, fig_height=15, fig_width=15):
    if flip_x_axis:
        image = np.fliplr(image)
    plt.rcParams['figure.figsize'] = [fig_width, fig_height]
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.title(title)
    plt.colorbar()
    plt.show()
    
def show_iter_solution(x):
    l = np.sqrt(x.shape[0]).astype(np.uintc)
    plt.imshow(np.fliplr(x.reshape(l, l)), cmap=plt.get_cmap("gray"))
    plt.colorbar()
    plt.show()
    
def show_sparse_matrix(image, title, fig_height=15, fig_width=15):
    plt.rcParams['figure.figsize'] = [fig_width, fig_height]
    plt.spy(image, markersize=5)
    plt.title(title)
    plt.show()
    
def normalise(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def rrmse(observed, ideal, decimal=6):
    return "{:.{}f}".format(np.sqrt((1 / observed.shape[0]**2) * np.sum((observed-ideal)**2) / np.sum(ideal**2)) * 100.0, decimal)

def laplacian_of_gaussian(x, y, sigma):
    p = (x**2.0 + y**2.0) / 2.0 * sigma**2.0
    return -(1.0 / (np.pi * sigma**4.0)) * (1.0 - p) * np.exp(-p)

# Generates a sparse decimation matrix using the Block Sparse Row matrix format (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html#scipy-sparse-bsr-matrix)
def decimation_matrix(original_dim, downsample_factor):
    
    if original_dim % downsample_factor != 0:
        raise ValueError(f"Downsample factor {downsample_factor} is not a valid factor of your matrix dimension {original_dim}.")
    if downsample_factor == original_dim:
        raise ValueError(f"Downsample factor {downsample_factor} cannot be the same as your matrix dimension {original_dim}.")
    if downsample_factor == 1: # effectively, no downsampling
        return np.identity(original_dim**2)
    # Otherwise assumed you want to downsample by a valid factor of original_dim...
    
    sampling_regions_per_dim = original_dim // downsample_factor
    samples_per_region_dim = downsample_factor
    # print(f"Sampling regions per dimension: {sampling_regions_per_dim}")
    # print(f"Samples per dimension: {samples_per_region_dim}")
    non_zero_entries = sampling_regions_per_dim**2 * samples_per_region_dim**2
    # print(f"Non-zero entries: {non_zero_entries}")
    
    rows = np.zeros(non_zero_entries, dtype=np.uintc)   # stores row indices for non-zero compressed sparse matrix entries
    cols = np.zeros(non_zero_entries, dtype=np.uintc)   # stores col indices for non-zero compressed sparse matrix entries
    vals = np.ones(non_zero_entries, dtype=np.ubyte)  # stores element value at [row, col] for non-zero entries
    
    # Generates linear x,y index strides for downsampling
    sample_stride_1D = np.arange(0, original_dim, downsample_factor)
    # print(sample_stride_1D)
    mesh = np.array(np.meshgrid(sample_stride_1D, sample_stride_1D))
    sample_strides_2D = mesh.T.reshape(-1, 2)
  
    neighbour_strides_1D = np.arange(samples_per_region_dim)
    neighbour_mesh = np.array(np.meshgrid(neighbour_strides_1D, neighbour_strides_1D))

    for index in np.arange(sample_strides_2D.shape[0]):
        neighbour_coords = neighbour_mesh.T.reshape(-1, 2) + sample_strides_2D[index] # generates (row, col) index pair for the nxn neighbours of each sampling point in sample_strides_2D
        neighbour_coords[:, 0] *= original_dim # scale y coord by high-resolution image dim to enable row striding (due to column-vector matrix flattening)
        neighbour_coords = np.sum(neighbour_coords, axis=1) # combine x and y coord into single array index
        rows[index * neighbour_coords.shape[0] : (index + 1) * neighbour_coords.shape[0]] = index
        cols[index * neighbour_coords.shape[0] : (index + 1) * neighbour_coords.shape[0]] = neighbour_coords
        
    return csr_matrix((vals, (rows, cols)))

def conv_matrix(l, kernel):
    k_supp = kernel.shape[0]
    k_half_supp = (k_supp-1)//2
    k_samples = k_supp**2
    col_offsets = np.repeat(np.arange(k_supp) - k_half_supp, k_supp) * (l - k_supp)
    diagonal_offsets = (np.arange(k_samples) - (k_samples-1)//2) + col_offsets
    m = diags(kernel.flatten(), diagonal_offsets, shape=(l**2, l**2), format="bsr", dtype=np.float32)
    
    mask_vals = np.repeat(1.0, k_supp)
    mask_offsets = np.linspace(-k_half_supp, k_half_supp, k_supp, dtype=np.intc)
    mask = diags(mask_vals, mask_offsets, shape=(l, l), format="bsr", dtype=np.float32)
    
    try:
        Parallel(n_jobs=-1, prefer='threads')(delayed(apply_mask)(row, l, k_half_supp, m, mask) for row in range(l))
    except Exception as e:
        print(e)
        
    return m  

def apply_mask(r, l, k_half_supp, m, mask):
    col_chunk_min = max(0, r-k_half_supp-1)
    col_chunk_max = min(l-1, r+k_half_supp+1)
    for c in range(col_chunk_min, col_chunk_max+1): # need to process only partial column chunks, probably needs parallelisation...
        m[r*l:r*l+l, c*l:c*l+l] = m[r*l:r*l+l, c*l:c*l+l].multiply(mask)
    
def generate_d_matrix(l, decimation_factor):
    return decimation_matrix(l, decimation_factor)
    
def generate_h_matrix(l, kernel):
    h = conv_matrix(l, kernel)
    # print(f"H matrix memory usage: {matrix_memory(h)}MB")
    # print(f"H matrix density: {matrix_density(h)}%")
    return h 
    
def generate_s_matrix(l, kernel=None, use_laplacian_of_gaussian=False):

    # default to 3x3 laplacian
    if kernel is None:
        # kernel = np.array([[0, -1,  0], [-1,  4, -1], [0, -1,  0]], dtype=np.float32)
        kernel = np.array([[-1, -1,  -1], [-1,  8, -1], [-1, -1,  -1]], dtype=np.float32)
    
    # Sharpening matrix (laplacian)
    if use_laplacian_of_gaussian:
        psf_central_support = 15 # num pixels around centre of PSF which represent approx 1/3 of the curve, maybe a bit bigger for padding
        downsampling = np.linspace(-(psf_central_support-1)//2, (psf_central_support-1)//2, num=psf_central_support)
        lap_of_gauss = np.zeros((downsampling.shape[0], downsampling.shape[0]), np.float32)
        lap_gauss_σ = 2.205128205 # need to play around with this to get similar structure to true laplacian 3x3
        for i in np.arange(downsampling.shape[0]):
            for j in np.arange(downsampling.shape[0]):
                lap_of_gauss[i][j] = -laplacian_of_gaussian(downsampling[i], downsampling[j], lap_gauss_σ)

        kernel = lap_of_gauss / np.max(lap_of_gauss)

        plt.plot(lap_of_gauss[lap_of_gauss.shape[0]//2])
        plt.show()

        # β *= 4.0

    s = conv_matrix(l, kernel)
    # print(type(s))
    # print(f"S matrix memory usage: {matrix_memory(s)}MB")
    # print(f"S matrix density: {matrix_density(s)}%")
    # show_sparse_matrix(s, "S")
    
    return s
    
def generate_b_image(l, n, w, dh, y):
    b = np.zeros(l**2, dtype=np.float32)

    for i in np.arange(n):
        b += w[i] * csr_matrix.transpose(dh) @ y[i].flatten()
        
    # print((b.size * b.itemsize) / 10**6)
    
    return b

def gaussian_2d(kernlen=21, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d * kern2d.sum()

### ===================================================================================================================
### FUNCTIONS FOR PRODUCING DH MATRIX (START)
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

@njit(parallel = True)
def populate_dh_buffers_batched(d_origins, kernel, low_res_dim, high_res_dim, original_kernel_samples, batch_size, batch_offset):
    
    kernel_samples = kernel.shape[0]
    kernel_offset = ((original_kernel_samples - 1) // 2) * (high_res_dim + 1)
    kernel = kernel.flatten()
    
    buffer_strides = np.zeros(batch_size, dtype=np.uintc)
    left_clip = np.zeros(batch_size, dtype=np.uintc)
    right_clip = np.zeros(batch_size, dtype=np.uintc)
    
    # First pass to determine how big to make row/col/val buffers and the stride used for each thread to populate each respective row of DH
    for i in prange(batch_size):
        repeated_range = np.repeat(np.arange(kernel_samples).astype(np.intc), kernel_samples)
        cols = repeated_range.reshape(-1, kernel_samples).T.flatten() + (repeated_range * high_res_dim) + d_origins[i+batch_offset] - kernel_offset
        left_clip[i] = cols[cols < 0].shape[0]
        right_clip[i] = cols[cols >= high_res_dim**2].shape[0]
    
    samples_per_clipped_dh_row = kernel_samples**2 - (left_clip + right_clip)
    total_samples = samples_per_clipped_dh_row.sum()
    
    # Below for Numba annotated func, cumsum doesnt support type...
    buffer_strides = np.append([0], np.cumsum(samples_per_clipped_dh_row))
    # Below for regular func, cumsum requires dtype...
    # buffer_strides = np.append([0], np.cumsum(samples_per_clipped_dh_row, dtype=np.uintc)) # prepend 0 to allow for ranges
    
    row_buffer = np.zeros(total_samples, dtype=np.uintc)
    col_buffer = np.zeros(total_samples, dtype=np.uintc)
    val_buffer = np.zeros(total_samples, dtype=np.float32)
    
    # Second pass to populate the row/col/val buffers using predetermined strides and clipping parameters
    for i in prange(batch_size):
        repeated_range = np.repeat(np.arange(kernel_samples).astype(np.intc), kernel_samples)
        cols = repeated_range.reshape(-1, kernel_samples).T.flatten() + (repeated_range * high_res_dim) + d_origins[i+batch_offset] - kernel_offset
        row_buffer[buffer_strides[i] : buffer_strides[i+1]] = i+batch_offset
        col_buffer[buffer_strides[i] : buffer_strides[i+1]] = cols[left_clip[i] : kernel_samples**2 - right_clip[i]]
        val_buffer[buffer_strides[i] : buffer_strides[i+1]] = kernel[left_clip[i] : kernel_samples**2 - right_clip[i]]
        
    return row_buffer, col_buffer, val_buffer
        
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
    return np.tile(np.arange(0, high_res_dim, downsample), low_res_dim) + np.repeat(np.arange(low_res_dim) * high_res_dim * downsample, low_res_dim)

### ===================================================================================================================
### FUNCTIONS FOR PRODUCING DH MATRIX (END)
### ===================================================================================================================









# Attempting to build a solution which generates DH on the fly, instead of generating D and H independently (too memory intensive)
# def generate_dh_matrix(original_dim, downsample_factor, blur_kernel):
    
#     kernel_support = blur_kernel.shape[0]
#     print(f"K Supp: { blur_kernel.shape}")
#     stacked = np.zeros((blur_kernel.shape[0] + downsample_factor - 1, blur_kernel.shape[1] + downsample_factor - 1), dtype=np.float32)
#     stacked_support = stacked.shape[0] # assumed square
#     for r in range(downsample_factor):
#         for c in range(downsample_factor):
#             stacked[r:r+kernel_support, c:c+kernel_support] += blur_kernel
            
#     print(f"Stacked Supp: {stacked.shape}")
            
#     stacked_padded = np.pad(stacked, (original_dim - stacked_support) // 2)
#     # show_image(stacked_padded, "Stacked")
#     # show_image(stacked, "Stacked")
    
#     # All samples on row 0, i.e., flattened
#     stacked_rows = np.zeros(stacked_support**2, dtype=np.uintc)
#     # All samples on linear columns, where each 'row' is strided to simulate 2D coordinates
#     stacked_cols = np.arange(stacked_support**2) +  np.repeat(np.arange(stacked_support) * (original_dim - stacked_support), stacked_support)
    
#     stacked_coo = coo_array((stacked.flatten(), (stacked_rows, stacked_cols))).tocsr()
    
#     print(stacked_coo.shape)
    
#     samples_per_new_pixel = original_dim // downsample_factor
#     samples_per_new_pixel_per_dim = downsample_factor
    
#     sampling_start_indices = np.arange(0, samples_per_new_pixel * samples_per_new_pixel_per_dim, samples_per_new_pixel_per_dim)
#     # print(f"sampling_start_indices: {sampling_start_indices}")
#     sampling_start_indices = np.tile(sampling_start_indices, samples_per_new_pixel)
#     # print(f"sampling_start_indices: {sampling_start_indices}")
#     dh_column_start_indices = np.repeat(np.arange(samples_per_new_pixel) * original_dim * samples_per_new_pixel_per_dim, samples_per_new_pixel) - ((original_dim * ((kernel_support-1)//2)) + ((kernel_support-1)//2))
#     # print(f"dh_column_start_indices: {dh_column_start_indices}")
#     offset_strides = sampling_start_indices + dh_column_start_indices
    
#     # ATTEMPTING TO VECTORISE COO APPROACH
#     t_rows = np.arange(offset_strides.shape[0])
#     print(f"Rows: {t_rows}")
    
#     t_col_offsets = offset_strides[t_rows]
#     print(f"Col mins: {t_col_offsets}")
    
#     t_col_maxs = t_col_offsets + stacked_coo.shape[1]
#     print(f"Col maxs: {t_col_maxs}")
    
#     remove_lower_out_of_bounds = np.vectorize(lambda col_min, lower_bound : max(lower_bound, col_min))
#     t_col_min_trims = remove_lower_out_of_bounds(t_col_offsets, 0)
#     print(f"Col min trims: {t_col_min_trims}")
    
#     remove_upper_out_of_bounds = np.vectorize(lambda col_max, upper_bound : min(col_max, upper_bound))
#     t_col_max_trims = remove_upper_out_of_bounds(t_col_maxs, original_dim**2)
#     print(f"Col max trims: {t_col_max_trims}")
    
#     distance_out_of_bounds = np.vectorize(lambda origin, x : np.absolute(x - origin))
#     left_trims = distance_out_of_bounds(t_col_offsets, t_col_min_trims)
#     print(f"Stacked Left trims: {left_trims}")
          
#     right_trims = distance_out_of_bounds(t_col_maxs, t_col_max_trims)
#     print(f"Stacked Right trims: {stacked_coo.shape[1] - right_trims}")

#     # rows = np.arange(offset_strides.shape[0], dtype=np.uintc)
#     # cols = offset_strides
#     dists = np.subtract(t_col_max_trims, t_col_min_trims)
#     # print(f"Dists: {dists}")
#     non_zero_entries = np.sum(dists)
#     vals = np.zeros(non_zero_entries, dtype=np.float32)
#     rows = np.repeat(t_rows, np.subtract(t_col_max_trims, t_col_min_trims))
#     cols = np.zeros(non_zero_entries, dtype=np.float32)
#     strided_indices = np.insert(np.cumsum(dists), 0, 0)
    # print(strided_indices)
    
    # for r in range(offset_strides.shape[0]):
        # cols[strided_indices[r]:strided_indices[r+1]] = np.arange(t_col_min_trims[r], t_col_max_trims[r])
        # vals[strided_indices[r]:strided_indices[r+1]] = (stacked_coo[:, left_trims[r] : stacked_coo.shape[1] - right_trims[r]]).eliminate_zeros()
        # print((stacked_coo[:, left_trims[r] : stacked_coo.shape[1] - right_trims[r]]).eliminate_zeros())
        
    # print(cols)
    # print(vals)
    
    # END VECTORISATION
        
    # Populating one row of DH matrix
#     for row in range(offset_strides.shape[0]):
#         col_offset_current_row = offset_strides[row]
#         col_min = col_offset_current_row
#         col_max = col_min + stacked_coo.shape[1]
#         # print(f"Row: {row}, col min/max: {col_min}, {col_max}")
        
#         # work out how much of stacked_coo can be copied to mat[row], avoiding out of bounds indexing
#         col_min_trim = max(0, col_min)
#         col_max_trim = min(original_dim**2, col_max)

#         stacked_coo_left_trim = np.absolute(col_min_trim - col_min)
#         stacked_coo_right_trim = np.absolute(col_max_trim - col_max)
        
#         mat[row, col_min_trim : col_max_trim] = stacked_coo[:, stacked_coo_left_trim : stacked_coo.shape[1] - stacked_coo_right_trim]
    
    # return csr_matrix((vals, (rows, cols)), shape=(samples_per_new_pixel**2, original_dim**2))
    # return mat.tocsr()



# Attempting to produce dh matrix incrementally
# def generate_dh_matrix_batched(original_dim, downsample_factor, blur_kernel):
    
#     # First thing we need is a stacked blur kernel
#     kernel_dim = blur_kernel.shape[0]
#     print(f"K Dim: {kernel_dim}")
#     stacked = np.zeros((kernel_dim + downsample_factor - 1, kernel_dim + downsample_factor - 1), dtype=np.float32)
#     stacked_kernel_dim = stacked.shape[0] # assumed square
#     for r in range(downsample_factor):
#         for c in range(downsample_factor):
#             stacked[r:r+kernel_dim, c:c+kernel_dim] += blur_kernel
    
#     print(f"Stacked K Dim: {stacked_kernel_dim}")
#     # show_image(stacked, "Stacked")

#     stacked_rows = np.zeros(stacked_kernel_dim**2, dtype=np.uintc)
#     stacked_cols = np.arange(stacked_kernel_dim**2) +  np.repeat(np.arange(stacked_kernel_dim) * (original_dim - stacked_kernel_dim), stacked_kernel_dim)
#     stacked_coo = coo_array((stacked.flatten(), (stacked_rows, stacked_cols))).tocsr()
#     print(stacked_coo.data)
    
#     # Helper funcs
#     mask_lower_out_of_bounds = np.vectorize(lambda col_min, lower_bound : max(lower_bound, col_min))
#     mask_upper_out_of_bounds = np.vectorize(lambda col_max, upper_bound : min(col_max, upper_bound))
#     dist = np.vectorize(lambda origin, x : np.absolute(x - origin))
    
#     samples_per_new_pixel = original_dim // downsample_factor
#     samples_per_new_pixel_per_dim = downsample_factor
#     # Need to rework the following to make it modular for batching
#     sampling_start_indices = np.arange(0, samples_per_new_pixel * samples_per_new_pixel_per_dim, samples_per_new_pixel_per_dim)
#     print(f"sampling_start_indices: {sampling_start_indices}")
#     sampling_start_indices = np.tile(sampling_start_indices, samples_per_new_pixel)
#     print(f"sampling_start_indices: {sampling_start_indices}")
#     dh_column_start_indices = np.repeat(np.arange(samples_per_new_pixel) * original_dim * samples_per_new_pixel_per_dim, samples_per_new_pixel) - ((original_dim * ((kernel_dim-1)//2)) + ((kernel_dim-1)//2))
#     print(f"dh_column_start_indices: {dh_column_start_indices}")
#     offset_strides = sampling_start_indices + dh_column_start_indices
    
#     # Define the base sparse matrix for accumulating into over time
#     dh = coo_matrix((samples_per_new_pixel**2, original_dim**2), dtype=np.float32)
    
#     rows_per_batch = samples_per_new_pixel**2
#     for batch in np.arange(0, samples_per_new_pixel**2, rows_per_batch):
        
#         working_rows = np.arange(batch, batch + rows_per_batch)
#         print(f"Working rows: {working_rows}")
        
#         working_cols_min = offset_strides[working_rows]
#         print(f"Working cols minimums: {working_cols_min}")
        
#         working_cols_max = working_cols_min + stacked_coo.shape[1]
#         print(f"Working cols maximums: {working_cols_max}")
        
#         working_cols_min_masked = mask_lower_out_of_bounds(working_cols_min, 0)
#         print(f"Working cols minimums masked: {working_cols_min_masked}")
        
#         working_cols_max_masked = mask_upper_out_of_bounds(working_cols_max, original_dim**2)
#         print(f"Working cols maximums masked: {working_cols_max_masked}")
        
#         left_trims = dist(working_cols_min, working_cols_min_masked)
#         print(f"Stacked Left trims: {left_trims}")
          
#         right_trims = dist(working_cols_max, working_cols_max_masked)
#         print(f"Stacked Right trims: {stacked_coo.shape[1] - right_trims}")
        
#         # rows = ...
#         # cols = ...
#         # vals = ...
        
#         # Accumulate partial to running dh
#         # dh +=  coo_matrix((vals, (rows, cols)), (samples_per_new_pixel**2, original_dim**2), dtype=np.float32)
        
    
    
#     # show_image(dh.todense(), "DH")
#     return dh.tocsr()

# def generate_h_matrix_tiled(high_res_dim, kernel, h_height, h_width, offset):
    
#     k_supp = kernel.shape[0]
#     k_half_supp = (k_supp-1)//2
#     k_samples = k_supp**2
    
#     col_offsets = np.repeat(np.arange(k_supp) - k_half_supp, k_supp) * (high_res_dim - k_supp)
#     # print(f"Column offsets: {col_offsets}")
    
#     diag_offsets = (np.arange(k_samples) - (k_samples-1)//2) + col_offsets - offset
#     # print(f"Diagonal offsets: {diag_offsets}")
    
#     diag_offsets_negative = diag_offsets[diag_offsets < 0]
#     # print(f"Negative diag offsets: {diag_offsets_negative}")
#     diag_offsets_negative_indices = np.where(np.absolute(diag_offsets_negative) > h_height)[0]
#     # print(f"Negative diag offsets indices: {diag_offsets_negative_indices}")
    
#     left_clip = diag_offsets_negative_indices.shape[0]
#     # print(f"Left clip: {left_clip}")
    
#     diag_offsets_positive = diag_offsets[diag_offsets > 0]
#     # print(f"Pos diag offsets: {diag_offsets_positive}")
#     diag_offsets_positive_indices = np.where(np.absolute(diag_offsets_positive) > h_width)[0]
#     # print(f"Pos diag offsets indices: {diag_offsets_positive_indices}")
#     right_clip = diag_offsets_positive_indices.shape[0]
#     # print(f"Right clip: {right_clip}")
    
#     flattened_clipped_kernel = kernel.flatten()[left_clip : k_samples - right_clip]
#     # print(f"Flattened clipped kernel: {flattened_clipped_kernel}")
    
#     diag_offsets = diag_offsets[left_clip : k_samples - right_clip]
#     # print(f"Diagonal offsets: {diag_offsets}")
    
#     if diag_offsets.size == 0: # empty, nothing represented in this region of h
#         return csr_matrix((h_height, h_width), dtype=np.float32)
#     else:
#         return diags(flattened_clipped_kernel, diag_offsets, shape=(h_height, h_width), format="lil", dtype=np.float32)
        
# # Attempting to produce DH on the fly by performing batched matmul using tiles
# def generate_dh_matrix_batched(high_res_dim, downsample_factor, kernel, tile_dim):

#     # Probably add some assertions here to make sure tile_dim fits nicely with this process
    
#     low_res_dim = high_res_dim // downsample_factor
#     samples_per_new_pixel_per_dim = downsample_factor
    
#     # Get D matrix ready, this scales very nicely with memory but maybe not runtime, review later if needed
#     d = generate_d_matrix(high_res_dim, downsample_factor)
    
#     # DH placeholder to be filled in using tiled approach
#     dh = lil_matrix((low_res_dim**2, high_res_dim**2), dtype=np.float32)
    
#     for row_tile in range(low_res_dim**2 // tile_dim):
#         for col_tile in range(high_res_dim**2 // tile_dim):
            
#             d_tile = d[row_tile * tile_dim : row_tile * tile_dim + tile_dim, :]
#             h_tile = generate_h_matrix_tiled(high_res_dim, kernel, high_res_dim**2, tile_dim, col_tile * tile_dim)
#             dh_tile = d_tile @ h_tile
            
#             if dh_tile.nnz == 0:
#                 continue # skip, as tile doesnt contribute to dh
            
#             dh[row_tile * tile_dim : row_tile * tile_dim + tile_dim, col_tile * tile_dim : col_tile * tile_dim + tile_dim] += dh_tile
            
#             # show_image(dh.todense(), "DH Intermediate")

#     return dh
    
# Attempting to producr DH on the fly using Numba for acceleration...
# @jit(nopython=True)
# def generate_dh_matrix_numba(high_res_dim, downsample_factor, kernel):
    
#     low_res_dim = high_res_dim // downsample_factor
    
#     kernel_dim = kernel.shape[0]
#     stacked = np.zeros((kernel_dim + downsample_factor - 1, kernel_dim + downsample_factor - 1), dtype=np.float32)
#     for r in range(downsample_factor):
#         for c in range(downsample_factor):
#             stacked[r:r+kernel_dim, c:c+kernel_dim] += kernel
            
#     # Storage mediums for row/col/vals of the dh matrix in coo format
#     row_indices = np.zeros(stacked.shape[0]**2 * low_res_dim**2, dtype=np.uintc)
#     col_indices = np.zeros(stacked.shape[0]**2 * low_res_dim**2, dtype=np.uintc)
#     dh_vals = np.zeros(stacked.shape[0]**2 * low_res_dim**2, dtype=np.float32)
                           
#     # Trying to work out where to put values...
#     sampling_start_indices = np.arange(0, low_res_dim * downsample_factor, downsample_factor)
#     # print(f"sampling_start_indices: {sampling_start_indices}")
#     sampling_start_indices = np.tile(sampling_start_indices, low_res_dim)
#     # print(f"sampling_start_indices: {sampling_start_indices}")
#     dh_column_start_indices = np.repeat(np.arange(low_res_dim) * original_dim * downsample_factor, low_res_dim) - ((original_dim * ((kernel_support-1)//2)) + ((kernel_support-1)//2))
#     # print(f"dh_column_start_indices: {dh_column_start_indices}")
#     offset_strides = sampling_start_indices + dh_column_start_indices
        
#     return coo_matrix((dh_vals, (row_indices, col_indices))).tocsr()
    
#     print(f"Stacked Supp: {stacked.shape}")
            
#     stacked_padded = np.pad(stacked, (original_dim - stacked_support) // 2)
#     # show_image(stacked_padded, "Stacked")
#     # show_image(stacked, "Stacked")
    
#     # All samples on row 0, i.e., flattened
#     stacked_rows = np.zeros(stacked_support**2, dtype=np.uintc)
#     # All samples on linear columns, where each 'row' is strided to simulate 2D coordinates
#     stacked_cols = np.arange(stacked_support**2) +  np.repeat(np.arange(stacked_support) * (original_dim - stacked_support), stacked_support)
    
#     stacked_coo = coo_array((stacked.flatten(), (stacked_rows, stacked_cols))).tocsr()
    
#     print(stacked_coo.shape)
    
#     samples_per_new_pixel = original_dim // downsample_factor
#     samples_per_new_pixel_per_dim = downsample_factor
    
#     sampling_start_indices = np.arange(0, samples_per_new_pixel * samples_per_new_pixel_per_dim, samples_per_new_pixel_per_dim)
#     # print(f"sampling_start_indices: {sampling_start_indices}")
#     sampling_start_indices = np.tile(sampling_start_indices, samples_per_new_pixel)
#     # print(f"sampling_start_indices: {sampling_start_indices}")
#     dh_column_start_indices = np.repeat(np.arange(samples_per_new_pixel) * original_dim * samples_per_new_pixel_per_dim, samples_per_new_pixel) - ((original_dim * ((kernel_support-1)//2)) + ((kernel_support-1)//2))
#     # print(f"dh_column_start_indices: {dh_column_start_indices}")
#     offset_strides = sampling_start_indices + dh_column_start_indices
    
#     # ATTEMPTING TO VECTORISE COO APPROACH
#     t_rows = np.arange(offset_strides.shape[0])
#     print(f"Rows: {t_rows}")
    
#     t_col_offsets = offset_strides[t_rows]
#     print(f"Col mins: {t_col_offsets}")
    
#     t_col_maxs = t_col_offsets + stacked_coo.shape[1]
#     print(f"Col maxs: {t_col_maxs}")
    
#     remove_lower_out_of_bounds = np.vectorize(lambda col_min, lower_bound : max(lower_bound, col_min))
#     t_col_min_trims = remove_lower_out_of_bounds(t_col_offsets, 0)
#     print(f"Col min trims: {t_col_min_trims}")
    
#     remove_upper_out_of_bounds = np.vectorize(lambda col_max, upper_bound : min(col_max, upper_bound))
#     t_col_max_trims = remove_upper_out_of_bounds(t_col_maxs, original_dim**2)
#     print(f"Col max trims: {t_col_max_trims}")
    
#     distance_out_of_bounds = np.vectorize(lambda origin, x : np.absolute(x - origin))
#     left_trims = distance_out_of_bounds(t_col_offsets, t_col_min_trims)
#     print(f"Stacked Left trims: {left_trims}")
          
#     right_trims = distance_out_of_bounds(t_col_maxs, t_col_max_trims)
#     print(f"Stacked Right trims: {stacked_coo.shape[1] - right_trims}")

    # rows = np.arange(offset_strides.shape[0], dtype=np.uintc)
    # cols = offset_strides
    # dists = np.subtract(t_col_max_trims, t_col_min_trims)
    # print(f"Dists: {dists}")
    # non_zero_entries = np.sum(dists)
    # vals = np.zeros(non_zero_entries, dtype=np.float32)
    # rows = np.repeat(t_rows, np.subtract(t_col_max_trims, t_col_min_trims))
    # cols = np.zeros(non_zero_entries, dtype=np.float32)
    # strided_indices = np.insert(np.cumsum(dists), 0, 0)
    # print(strided_indices)
    
    # for r in range(offset_strides.shape[0]):
        # cols[strided_indices[r]:strided_indices[r+1]] = np.arange(t_col_min_trims[r], t_col_max_trims[r])
        # vals[strided_indices[r]:strided_indices[r+1]] = (stacked_coo[:, left_trims[r] : stacked_coo.shape[1] - right_trims[r]]).eliminate_zeros()
        # print((stacked_coo[:, left_trims[r] : stacked_coo.shape[1] - right_trims[r]]).eliminate_zeros())
        
    # print(cols)
    # print(vals)
    
    # END VECTORISATION
        
    # Populating one row of DH matrix
#     for row in range(offset_strides.shape[0]):
#         col_offset_current_row = offset_strides[row]
#         col_min = col_offset_current_row
#         col_max = col_min + stacked_coo.shape[1]
#         # print(f"Row: {row}, col min/max: {col_min}, {col_max}")
        
#         # work out how much of stacked_coo can be copied to mat[row], avoiding out of bounds indexing
#         col_min_trim = max(0, col_min)
#         col_max_trim = min(original_dim**2, col_max)

#         stacked_coo_left_trim = np.absolute(col_min_trim - col_min)
#         stacked_coo_right_trim = np.absolute(col_max_trim - col_max)
        
#         mat[row, col_min_trim : col_max_trim] = stacked_coo[:, stacked_coo_left_trim : stacked_coo.shape[1] - stacked_coo_right_trim]
    
    # return csr_matrix((vals, (rows, cols)), shape=(samples_per_new_pixel**2, original_dim**2))
    # return mat.tocsr()