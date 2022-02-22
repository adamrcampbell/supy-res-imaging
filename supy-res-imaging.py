
import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d
from skimage.filters import gaussian
from skimage import data, io, color
from skimage.exposure import rescale_intensity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import sys

#======================================================================
# Preliminary implementation of the Iterative Back Projection technique
#======================================================================

sample = color.rgb2gray(data.astronaut())
# resampling method for up/down-scaling images
# note: 0 = nearest, 1 = bilinear, 2 = cubic
resampling = 2
# number of y low resolution images
num_low_res_images = 5
# assumed to be square
low_res_image_dim = 256
high_res_image_dim = sample.shape[0]
image_ratio = high_res_image_dim / low_res_image_dim

# error_threshold = 10.0
max_iter = 100
error = sys.float_info.max
has_converged = False

low_res = np.ndarray((num_low_res_images, low_res_image_dim, low_res_image_dim), np.float32)
for k in range(num_low_res_images):
    low_res[k] = scipy.ndimage.zoom(gaussian(sample), (1/image_ratio, 1/image_ratio), order=resampling)

# resize low res images to same dimensions as high res image
low_res_upsampled = scipy.ndimage.zoom(low_res, (1, image_ratio, image_ratio), order=resampling)
# print(low_res_upsampled)

# Synthesize initial high resolution image guess, should this be sum or average of pixels? idk
high_res = np.average(low_res_upsampled, axis=0)

sharpen_kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                           [0, -1, 0]])

current_iter = 0
while current_iter < max_iter and not has_converged:
    current_iter += 1
    print("Processing iteration %d..." % current_iter)

    # Predict low resolution images from high resolution image
    # note: probably a terrible way to do it atm...
    low_res_predicted = np.ndarray((num_low_res_images, low_res_image_dim, low_res_image_dim), np.float32)
    for k in range(num_low_res_images):
        low_res_predicted[k] = scipy.ndimage.zoom(gaussian(high_res), (1/image_ratio, 1/image_ratio), order=resampling)

    # Calculate the error between low_res and low_res_predicted
    l2norm_sum = 0
    for k in range(num_low_res_images):
        l2norm_sum = l2norm_sum + np.linalg.norm(low_res[k] - low_res_predicted[k])
    curr_err = np.sqrt(1/num_low_res_images * l2norm_sum)

    if curr_err > error:
        has_converged = True
        break
    
    error = curr_err
    print("Error: %f..." % error)

    # Make high resolution version of low res predicted
    low_res_predicted_upsampled = scipy.ndimage.zoom(low_res_predicted, (1, image_ratio, image_ratio), order=resampling)
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
    weight = max(min(error, 1.0), 0.0) # clamp func for at most 1.0 multiplier
    print("Iter %d weight: %f" % (current_iter, weight))
    high_res_predicted = high_res + sum_diff * 1/num_low_res_images * weight
    # io.imshow(np.absolute(high_res), plugin="matplotlib", **{"cmap": "Greys"})
    # io.show()
    np.copyto(high_res, high_res_predicted)
    rmse = mean_squared_error(sample, high_res, squared=False)
    print("Iter %d rrmse %f...\n" % (current_iter, rmse * 100))
 
# Image rendering
# fig, axes = plt.subplots(1, 3)
# ax = axes.ravel()

# ax[0].imshow(sample, cmap=plt.cm.gray)
# ax[1].imshow(high_res, cmap=plt.cm.gray)
# ax[2].imshow(np.absolute(sample - high_res), cmap=plt.cm.gray)

# 
# plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3)
im1 = axes[0].imshow(sample, cmap=plt.cm.gray)
axes[0].set_title("Original (sum %.3f, min %.3f, max %.3f)"  % (np.sum(sample), np.min(sample), np.max(sample)))
im2 = axes[1].imshow(high_res, cmap=plt.cm.gray)
axes[1].set_title("Reconstructed (sum %.3f, min %.3f, max %.3f)" % (np.sum(high_res), np.min(high_res), np.max(high_res)))
abs_diff = np.absolute(sample - high_res)
im3 = axes[2].imshow(abs_diff, cmap=plt.cm.gray)
axes[2].set_title("Absolute Diff (sum %.3f, min %.3f, max %.3f)" % (np.sum(abs_diff), np.min(abs_diff), np.max(abs_diff)))
plt.show()

print("Finished processing...")