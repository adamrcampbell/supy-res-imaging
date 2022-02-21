
import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d
from skimage.filters import gaussian
from skimage import data, io

import sys

#======================================================================
# Preliminary implementation of the Iterative Back Projection technique
#======================================================================

# number of y low resolution images
# (assumed linear in time, aka snapshots)
num_low_res_images = 3
# assumed to be square
low_res_image_dim = 16
high_res_image_dim = 32
image_ratio = high_res_image_dim / low_res_image_dim

error_threshold = 10.0
max_iter = 10
error = sys.float_info.max

total_low_res_pixels = num_low_res_images * low_res_image_dim * low_res_image_dim
# Initial estimate of low resolution images, assumed we get this provided to us
low_res = np.arange(0, total_low_res_pixels, dtype=np.float32)
# Rejig memory for fit ZYX axis layout
low_res = np.reshape(low_res, (num_low_res_images, low_res_image_dim, low_res_image_dim))

# 2D interpolation across YX axis, ignoring Z interpolation
# note: order 0 = nearest, 1 = bilinear, 2 = cubic
# resize low res images to same dimensions as high res image
low_res_upsampled = scipy.ndimage.zoom(low_res, (1, image_ratio, image_ratio), order=0)
# print(low_res_upsampled)

# Synthesize initial high resolution image guess, should this be sum or average of pixels? idk
high_res = np.average(low_res_upsampled, axis=0)
# print(high_res)

current_iter = 0
while current_iter < max_iter:
    current_iter += 1
    print("Processing iteration %d..." % current_iter)

    # Predict low resolution images from high resolution image
    # note: probably a terrible way to do it atm...
    low_res_predicted = np.ndarray((num_low_res_images, low_res_image_dim, low_res_image_dim), np.float32)
    for k in range(num_low_res_images):
        low_res_predicted[k] = scipy.ndimage.zoom(gaussian(high_res), (1/image_ratio, 1/image_ratio), order=0)

    # Calculate the error between low_res and low_res_predicted
    l2norm_sum = 0
    for k in range(num_low_res_images):
        l2norm_sum = l2norm_sum + np.linalg.norm(low_res[k] - low_res_predicted[k])
    error = np.sqrt(1/num_low_res_images * l2norm_sum)
    print("Error: %f..." % error)

    sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

    # Make high resolution version of low res predicted
    low_res_predicted_upsampled = scipy.ndimage.zoom(low_res_predicted, (1, image_ratio, image_ratio), order=0)
    # Take the difference of low resolution upsampled and low resolution predicted upsampled
    low_res_upsampled_diff = low_res_upsampled - low_res_predicted_upsampled

    # Convolve each image with sharpen filter
    for k in range(num_low_res_images):
        output = convolve2d(low_res_upsampled_diff[k], sharpen_kernel)
        # trim off filter padding
        low_res_upsampled_diff[k] = output[1:high_res_image_dim+1, 1:high_res_image_dim+1]

    # Predict high resolution image
    high_res_predicted = np.zeros_like(high_res)
    sum_diff = np.sum(low_res_upsampled_diff, axis=0)
    high_res_predicted = high_res + sum_diff * 1/num_low_res_images
    np.copyto(high_res, high_res_predicted)
    
io.imshow(high_res)
io.show()
print("Finished processing...")
sys.exit()