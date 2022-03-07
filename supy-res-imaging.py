
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from skimage.filters import gaussian
from skimage import data, io, color
from skimage.util import random_noise, img_as_float32
from skimage.exposure import rescale_intensity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import sys

def display_image_stack(*images):
    fig, axes = plt.subplots(nrows=1, ncols=len(images))

    if len(images) == 1:
        im = axes.imshow(images[0], cmap=plt.cm.gray)
    else:
        for i in range(len(images)):
            im = axes[i].imshow(images[i], cmap=plt.cm.gray)
            axes[i].set_title("sum %.3f, min %.3f, max %.3f"  % (np.sum(images[i]), np.min(images[i]), np.max(images[i])))
    plt.show()

def display_image_stack_block(count, *images):
    fig, axes = plt.subplots(nrows=1, ncols=count)
    print(images[0].shape)

    for i in range(count):
        im = axes[i].imshow(images[0][i], cmap=plt.cm.gray)
        axes[i].set_title("sum %.3f, min %.3f, max %.3f"  % (np.sum(images[0][i]), np.min(images[0][i]), np.max(images[0][i])))
    plt.show()

def plot_error_over_time(timesteps, error, rrmse):
    plt.plot(range(timesteps), np.log2(error), label="Error (y_true, y_predict")
    # plt.plot(range(timesteps), rrmse, label="RRMSE (x_true, x_predict_k)")
    plt.xticks(range(timesteps))
    plt.xlabel("Timestep (t)")
    plt.ylabel("Error")
    plt.show()

#======================================================================
# Preliminary implementation of the Iterative Back Projection technique
#======================================================================

# x_true = color.rgb2gray(data.astronaut())
x_true = img_as_float32(data.camera())
# number of y low resolution images
num_y_samples = 10
# controls the size of low resolution image dimensions
downsample_factor = 0.5
x_img_dim = x_true.shape[0]
y_img_dim = int(x_img_dim * downsample_factor)
upsample_factor = x_img_dim / y_img_dim

max_iter = 100

y_true = np.ndarray((num_y_samples, y_img_dim, y_img_dim), np.float32)
for k in range(num_y_samples):
    # Smooth image, downsample, apply gaussian noise
    y_true[k] = random_noise(zoom(gaussian(x_true), downsample_factor))

# Predict initial estimate of x_true
x_predict = zoom(np.average(y_true, axis=0), upsample_factor)

laplace_scale = 1.0
# laplace_deblur = np.array([[0, -1,  0], [-1,  5, -1], [0, -1,  0]], dtype=np.float32)
laplace_deblur = np.array([[0, -laplace_scale,  0], [-laplace_scale,  1.0 + 4*laplace_scale, -laplace_scale], [0, -laplace_scale,  0]])
# laplace_deblur = np.array([[0.0, 0.0,  0.0], [0.0,  1.0, 0.0], [0.0, 0.0,  0.0]])
delta = np.array([[0.0, 0.0,  0.0], [0.0,  1.0, 0.0], [0.0, 0.0,  0.0]])

# print(np.sum((delta - gaussian(laplace_deblur))**2))
# io.imshow(delta-gaussian(laplace_deblur))
# io.show()
# sys.exit()

x_estimates = np.ndarray((max_iter, x_img_dim, x_img_dim), np.float32)
x_error = np.zeros(max_iter)
x_rrmse = np.zeros(max_iter)

current_iter = 0
while current_iter < max_iter:
    print("Processing iteration %d..." % current_iter)

    # Predict low resolution images from high resolution image
    # note: probably a terrible way to do it atm...
    y_predict = np.ndarray((num_y_samples, y_img_dim, y_img_dim), np.float32)
    for k in range(num_y_samples):
        y_predict[k] = zoom(gaussian(x_predict), downsample_factor)

    # Calculate the error between low_res and low_res_predicted
    l2norm_sum_squares = 0.0
    for k in range(num_y_samples):
        l2norm_sum_squares = l2norm_sum_squares + np.linalg.norm(y_true[k] - y_predict[k])**2
    error = np.sqrt(1.0/num_y_samples * l2norm_sum_squares)
    # print("Iter %d error: %f..." % (current_iter, error))
    x_error[current_iter] = error

    # Make high resolution version of low res predicted
    y_true_pred_diff = zoom(y_true - y_predict, (1, int(upsample_factor), int(upsample_factor)))

    # Convolve each image with sharpen filter
    for k in range(num_y_samples):
        # convolve with laplacian deblur, trim edges (padded during convolution)
        y_true_pred_diff[k] = convolve2d(y_true_pred_diff[k], laplace_deblur)[1:x_img_dim+1, 1:x_img_dim+1]

    # Predict high resolution image
    x_new_predict = np.zeros_like(x_predict)
    sum_diff = np.sum(y_true_pred_diff, axis=0)
    # weight = max(min(error, 1.0), 0.0) # clamp func for at most 1.0 multiplier
    # weight = max(min(1.0/error, 1.0), 0.0)
    weight = 0.1
    # print("Iter %d weight: %f" % (current_iter, weight))
    x_new_predict = x_predict + sum_diff * 1.0/num_y_samples * weight
    rrmse = mean_squared_error(x_true, x_new_predict, squared=False) * 100.0
    # print("x_true min/max: %f, %f..." % (np.min(x_true), np.max(x_true)))
    # print("x_pred min/max: %f, %f..." % (np.min(x_new_predict), np.max(x_new_predict)))
    x_rrmse[current_iter] = rrmse
    # print("Iter %d rrmse %f...\n" % (current_iter, rrmse))
    np.copyto(x_predict, x_new_predict)
    np.copyto(x_estimates[current_iter], x_new_predict)
    current_iter += 1

display_image_stack(x_true, x_predict, np.absolute(x_true - x_predict), y_predict[0])
# display_image_stack_block(max_iter, x_estimates)
plot_error_over_time(max_iter, x_error, x_rrmse)

print("Finished processing...")