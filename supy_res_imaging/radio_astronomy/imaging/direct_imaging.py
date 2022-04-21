import numpy as np
from numba import njit

@njit( parallel = False )
def direct_imaging_w_correction(image, visibilities, uvw_coords, num_uvw_coords, num_channels, image_dim, cell_size_radians, channel_hz_start, bandwidth_increment):
    
    num_visibilities = num_uvw_coords * num_channels
    
    for r in range(image_dim):
        for c in range(image_dim):
           
            pixel_index = r * image_dim + c
            
            x = ((pixel_index % image_dim) - image_dim / 2) * cell_size_radians
            y = ((pixel_index // image_dim) - image_dim / 2) * cell_size_radians
            img_correction = np.sqrt(1.0 - x*x - y*y)
            w_correction = img_correction - 1.0
            pixel_sum = 0.0
            
            for v in range(num_visibilities):
                
                current_baseline = v // num_channels
                current_channel = v % num_channels
                m2w = (channel_hz_start + (bandwidth_increment * current_channel)) / 299792458.0
                uvw = uvw_coords[current_baseline] * m2w
                
                theta = 2.0 * np.pi * (x * uvw[0] + y * uvw[1] + w_correction * uvw[2])
                theta_vis_product = visibilities[v] * (np.cos(theta) + np.sin(theta) * 1.0j)
                pixel_sum += theta_vis_product.real
            
            image[r][c] += pixel_sum * img_correction