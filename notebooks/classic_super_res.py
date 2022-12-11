import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.stats import norm
from scipy.sparse import csc_matrix, dia_matrix, diags
from scipy.sparse.linalg import spsolve, factorized
from scipy.linalg import solve
from skimage import data, io, color
from skimage.transform import resize

plt.rcParams['figure.figsize'] = [10, 10]

def show_image(image, title, flip_x_axis=False):
    if flip_x_axis:
        image = np.fliplr(image)
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.title(title)
    plt.colorbar()
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
    vals = np.ones(non_zero_entries, dtype=np.float32)  # stores element value at [row, col] for non-zero entries
    
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
        
    return csc_matrix((vals, (rows, cols)))

def janky_conv_matrix(l, kernel):
    k_supp = kernel.shape[0]
    k_half_supp = (k_supp-1)//2
    k_samples = k_supp**2
    col_offsets = np.repeat(np.arange(k_supp) - k_half_supp, k_supp) * (l - k_supp)
    diagonal_offsets = (np.arange(k_samples) - (k_samples-1)//2) + col_offsets
    return dia_matrix.tocsc(diags(kernel.flatten(), diagonal_offsets, shape=(l**2, l**2)))

timesteps = 30 # total timesteps
timesteps_per_y = 1
l = 400
m = 100
n = timesteps // timesteps_per_y
w = np.ones(n)
β = 8.0

# False = use static 3x3 laplacian kernel
# True = use customisable laplacian of gaussian kernel
use_laplacian_of_gaussian = True

# all time steps direct image
# filename = "../data/direct_image_ts_0_29.bin"
# filename = "../data/direct_image_ts_0_29_800x800.bin"
filename = "../datasets/gleam_small/images/direct_image_ts_0_29_400x400.bin"
x_true = np.fromfile(filename, dtype=np.float32)
x_true = np.reshape(x_true, (l, l))
# x_true = resize(x_true.reshape(800, 800), (l, l), anti_aliasing=False, order=1)
x_true = normalise(x_true)
# show_image(x_true, "True X")

# x_orig = np.fromfile("../data/direct_image_ts_0_29.bin", dtype=np.float32)
# show_image(normalise(x_orig.reshape(l, l)), "Original True X (100^2)")

# all time steps direct psf
# filename = "../data/direct_psf_ts_0_29.bin"
# filename = "../data/direct_psf_ts_0_29_800x800.bin"
# x_psf = np.fromfile(filename, dtype=np.float32)
# x_psf = resize(x_psf.reshape(800, 800), (l, l), anti_aliasing=False, order=1)
# x_psf = x_psf.reshape(l, l)[1:, 1:] # trim row 0 and all column 0 (to ensure odd dimensions with the peak in the center)
# x_psf /= np.max(x_psf)
# show_image(x_psf, "PSF (800^2 averaged to 100^2)")

# filename = "../data/direct_psf_ts_0_29_800x800.bin"
filename = "../datasets/gleam_small/images/direct_psf_ts_0_29_400x400.bin"
x_psf = np.fromfile(filename, dtype=np.float32).reshape(400, 400)[1:, 1:]
# show_image(x_psf, "Untrimmed PSF")
x_psf = resize(x_psf, (l-1, l-1), anti_aliasing=False, order=1)
x_psf = np.pad(x_psf, ((1, 0), (1, 0))) # pad with new 0th row/col to ensure trimming from centre
# show_image(x_psf, "Untrimmed PSF - RESIZED")
# plt.plot(x_psf[x_psf.shape[0]//2])
# plt.show()

trim_half_len = 10
psf_min = l//2 - (trim_half_len - 1)
psf_max = l//2 + trim_half_len
psf_support = psf_max - psf_min
# print(psf_support)
# x_psf = x_psf.reshape(l, l)[psf_min:psf_max, psf_min:psf_max]
x_psf_trimmed = x_psf.copy()[psf_min:psf_max, psf_min:psf_max]
x_psf_trimmed /= np.sum(x_psf_trimmed)
# print(np.sum(x_psf_trimmed))
# plt.plot(x_psf[x_psf.shape[0]//2])
# plt.show()

# print(f"PSF Shape: {x_psf_trimmed.shape[0]}")
# show_image(x_psf_trimmed, "PSF")

print(f"l^2:                      {l**2}")
print(f"(psf_support - 2) * m**2: {(psf_support - 2) * m**2}")
print(f"n * m**2:                 {n * m**2}")

# Determine whether we have the right criteria to perform super-resolution imaging to obtain X
if l**2 <= np.minimum((psf_support - 2) * m**2, n * m**2):
    print("Super-resolution imaging technically possible, continuing...\n")
else:
    raise ValueError("High-resolution image not possible to reproduce from your set up, review your configuration against this if statement conditional.\n")

# Storing all low-res images as layered stack
y = np.zeros((n, m, m))

# batched time steps direct images
for i in np.arange(n):
    timestep_range_start = i * timesteps_per_y
    timestep_range_end = timestep_range_start + timesteps_per_y
    filename = f"../datasets/gleam_small/images/direct_image_ts_{timestep_range_start}_{timestep_range_end - 1}_{m}x{m}.bin"
    y[i] = np.fromfile(filename, dtype=np.float32).reshape(m, m)
    y[i] = normalise(y[i])
    # show_image(y[i], f"$Y {i}$")
    
# batched time steps point spread functions
# for i in np.arange(N):
#     start = i * timesteps_per_y
#     end = start + timesteps_per_y - 1
#     filename = "../data/direct_psf_ts_%d_%d.bin" % (start, end)
#     Y_i_psf = np.fromfile(filename, dtype=np.float32)
#     Y_i_psf = Y_i_psf.reshape(L, L)
#     # show_image(Y_i_psf, "$Y_{%d}$ PSF" % i)

# Decimation matrix
d = decimation_matrix(l, 4)
# d_numpy_size_bytes = d.size * d.itemsize
# print(f"D matrix uncompressed sparse: {d_numpy_size_bytes / 10**6} MB")
# d = bsr_matrix(d)
d_csr_matrix_bytes = d.data.nbytes + d.indptr.nbytes + d.indices.nbytes
print(f"D matrix compressed sparse: {d_csr_matrix_bytes / (10**6)} MB")
# print(f"D matrix reduced by {100 - (d_csr_matrix_bytes / d_numpy_size_bytes * 100)}%\n")

# Blur matrix (psf)
h = janky_conv_matrix(l, x_psf_trimmed)
h_csr_matrix_bytes = h.data.nbytes + h.indptr.nbytes + h.indices.nbytes
print(f"H matrix compressed sparse: {h_csr_matrix_bytes / (10**6)} MB")
# show_image(h.todense(), "H")

# Sharpening matrix (laplacian)
if use_laplacian_of_gaussian:
    psf_central_support = 60 # 15 # num pixels around centre of PSF which represent approx 1/3 of the curve, maybe a bit bigger for padding
    downsampling = np.linspace(-(psf_central_support-1)//2, (psf_central_support-1)//2, num=psf_central_support)
    lap_of_gauss = np.zeros((downsampling.shape[0], downsampling.shape[0]), np.float32)
    lap_gauss_σ = 2.205128205 # need to play around with this to get similar structure to true laplacian 3x3
    for i in np.arange(downsampling.shape[0]):
        for j in np.arange(downsampling.shape[0]):
            lap_of_gauss[i][j] = -laplacian_of_gaussian(downsampling[i], downsampling[j], lap_gauss_σ)

    lap_of_gauss /= np.max(lap_of_gauss)

    # plt.plot(lap_of_gauss[lap_of_gauss.shape[0]//2])
    # plt.show()

    s = janky_conv_matrix(l, lap_of_gauss)
    β *= 4.0
    # print(f"Laplacian Shape: {lap_of_gauss.shape[0]}")
else:
    laplacian = np.array([[0, -1,  0], [-1,  4, -1], [0, -1,  0]], dtype=np.float32)
    s = janky_conv_matrix(l, laplacian)

s_csr_matrix_bytes = s.data.nbytes + s.indptr.nbytes + s.indices.nbytes
print(f"S matrix compressed sparse: {s_csr_matrix_bytes / (10**6)} MB")
    
dh = d @ h

dh_csr_matrix_bytes = dh.data.nbytes + dh.indptr.nbytes + dh.indices.nbytes
print(f"DH matrix compressed sparse: {dh_csr_matrix_bytes / (10**6)} MB")
    
# Notes to self: seems like lap_gauss_σ sits around 1.41 for this example, any higher or lower results in worse RRMSE. Changing the psf_central_support makes no difference, and taking more samples (i.e. num)
# just means that lap_gauss_σ is multiplied by the factor of difference (i.e., double the num samples needs double the lap_gauss_σ). Also found that beta is best multiplied by 4.0, not 16 (the square).
# Not sure what to do from here... seems like the laplacian of gaussian doesnt really serve any beneficial purpose...

b = np.zeros(l**2, dtype=np.float32)

for i in np.arange(n):
    b += w[i] * csc_matrix.transpose(dh) @ y[i].flatten()
    
# show_image(normalise(b.reshape(l, l)), "B")

print((b.size * b.itemsize) / 10**6)
print(b.shape)

# lhs = β * np.matmul(s.T, s)
# rhs = (h.T @ d.T @ d @ h) * np.sum(w)
a = (β * s @ csc_matrix.transpose(s)) + (csc_matrix.transpose(dh) @ dh * np.sum(w))
derp = factorized(a)
print("Hello")

# a = bsr_matrix.toarray(a)
# show_image(a.reshape(l**2, l**2), "A")

print(type(derp))
# print((a.data.nbytes + a.indptr.nbytes + a.indices.nbytes) / 10**6)

# Solving via CPU and numpy...
# x = np.linalg.solve(a, b)
# x = spsolve(a, b) # "not enough memory to perform factorization"
# x = solve(a, b) # "Sparse matrices are not supported by this function. Perhaps one of the scipy.sparse.linalg functions would work instead"
x = derp(b)
x = x.reshape(l, l)
# x = normalise(x)

# b_true = np.matmul(normalise(a), x_true.flatten())
# x_accurate = np.linalg.solve(a, b_true)
# x_accurate = x_accurate.reshape(100, 100)

# show_image(normalise(x), "Solved X")
# print(f"RRMSE: B and True B -> {rrmse(normalise(b.reshape(l, l)), normalise(b_true.reshape(l, l)))}")
# print(f"L2 between B and True X: {np.sqrt(np.sum((normalise(b.reshape(l, l))-normalise(x))**2))}")
# show_image(normalise(b.reshape(l, l)), "B")
# show_image(normalise(b_true.reshape(l, l)), "True B")

# print(f"RRMSE: X and Accurate X -> {rrmse(normalise(x), normalise(x_accurate))}")
# print(f"RRMSE: Accurate X and True X -> {rrmse(normalise(x_accurate), normalise(x_true))}")
print(f"Beta: {β}")
print(f"RRMSE: Solved X and True X -> {rrmse(normalise(x), normalise(x_true))}")
print(f"True X peak location: {np.unravel_index(np.argmax(x_true, axis=None), x_true.shape)}")
print(f"Solved X peak location: {np.unravel_index(np.argmax(x, axis=None), x.shape)}")

# plt.plot(x_true[93, :], label="True X")
# plt.plot(x[93, :], label="Solved X")
# plt.legend()
# plt.show()

# Multiply A left hand side (beta and sharpening matrices) by X
# a_laplacian = lhs # np.matmul(lhs, x.flatten()).reshape(l, l)
# show_image((a_laplacian), "A-laplacian", flip_x_axis=True)
# a_lap_l1 = np.max(np.sum(np.absolute(a_laplacian), axis=0))
# a_lap_linf = np.max(np.sum(np.absolute(a_laplacian), axis=1))
# print(f"A_laplacian l1 norm: {a_lap_l1}")
# print(f"A_laplacian linf norm: {a_lap_linf}")

# a_deci_blur = rhs # np.matmul(rhs, x.flatten()).reshape(l, l)
# show_image((a_deci_blur), "A-deci-blur", flip_x_axis=True)
# a_dec_l1 = np.max(np.sum(np.absolute(a_deci_blur), axis=0))
# a_dec_linf = np.max(np.sum(np.absolute(a_deci_blur), axis=1))
# print(f"A_deci_blur l1 norm: {a_dec_l1}")
# print(f"A_deci_blur linf norm: {a_dec_linf}")

# print(f"A_deci_blur / A_laplacian l1 ratio: {a_dec_l1 / a_lap_l1}")
# print(f"A_deci_blur / A_laplacian linf ratio: {a_dec_linf / a_lap_linf}")

show_image(normalise(x), "Solved X", flip_x_axis=True)
# show_image(normalise(x_accurate[10:l-11, 10:l-11]), "Solved X using True X (IDFT)")
# show_image(normalise(x_accurate), "Solved X using True X (IDFT)")
show_image(normalise(x_true), "True X", flip_x_axis=True)
# show_image(normalise(a_laplacian + a_deci_blur), "A parts summed to B", flip_x_axis=True)
