import numpy as np
from PDE_find import Diff, Diff2, FiniteDiff
import Data_generator as Data_generator
import scipy.io as scio
from requests import get
from inspect import isfunction
import math
import pdb
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn import Linear,Tanh,Sequential
from torch.autograd import Variable
import configure as config
from configure import divide
from scipy.signal import windows

simple_mode = True
see_tree = None
plot_the_figures = False
use_metadata = False
use_difference = True

if use_difference == True:
    use_autograd = False
    print('Using difference method')
else:
    use_autograd = True
    print('Using autograd method')

def cubic(inputs):
    return np.power(inputs, 3)

# def divide(up, down, eta=1e-10):
#     while np.any(down == 0):
#         down += eta
#     return up/down

def get_random_int(max_int):
    random_result = get('https://www.random.org/integers/?num=1&min=0&max={0}&col=1&base=10&format=plain&rnd=new'.format(max_int)).content
    try:
        int(random_result)
    except:
        print(random_result)
    return int(random_result)

# rand = get_random_int(1e6)
rand = config.seed #0
print('random seed:{}'.format(rand))
# 237204
np.random.seed(rand)
random.seed(rand)

# load Metadata
u = Data_generator.u
x = Data_generator.x
t = Data_generator.t
x_all = Data_generator.x_all
n, m = u.shape
dx = x[2]-x[1]
dt = t[1]-t[0]
# 扩充维度使得与u的size相同
x = np.tile(x, (m, 1)).transpose((1, 0))
x_all = np.tile(x_all, (m, 1)).transpose((1, 0))
t = np.tile(t, (n, 1))

# load Origin data
u_origin=config.u
x_origin=config.x
t_origin=config.t
n_origin, m_origin = u_origin.shape
dx_origin = x_origin[2]-x_origin[1]
dt_origin = t_origin[1]-t_origin[0]
# 扩充维度使得与u的size相同
x_origin = np.tile(x_origin, (m_origin, 1)).transpose((1, 0))
t_origin = np.tile(t_origin, (n_origin, 1))

# 差分
# calculate the error of correct cofs & correct terms
if use_difference == True:
    ut = np.zeros((n, m))
    for idx in range(n):
        ut[idx, :] = FiniteDiff(u[idx, :], dt)
    ux = np.zeros((n, m))
    uxx = np.zeros((n, m))
    uxxx = np.zeros((n, m))
    for idx in range(m):
        ux[:, idx] = FiniteDiff(u[:, idx], dx) #idx is the id of one time step
    for idx in range(m):
        uxx[:, idx] = FiniteDiff(ux[:, idx], dx)
    for idx in range(m):
        uxxx[:, idx] = FiniteDiff(uxx[:, idx], dx)

    ut_origin = np.zeros((n_origin, m_origin))
    for idx in range(n_origin):
        ut_origin[idx, :] = FiniteDiff(u_origin[idx, :], dt_origin)
    ux_origin = np.zeros((n_origin, m_origin))
    uxx_origin = np.zeros((n_origin, m_origin))
    uxxx_origin = np.zeros((n_origin, m_origin))
    for idx in range(m_origin):
        ux_origin[:, idx] = FiniteDiff(u_origin[:, idx], dx_origin) #idx is the id of one time step
    for idx in range(m_origin):
        uxx_origin[:, idx] = FiniteDiff(ux_origin[:, idx], dx_origin)
    for idx in range(m_origin):
        uxxx_origin[:, idx] = FiniteDiff(uxx_origin[:, idx], dx_origin)


#### Add Noise ####
# avg = np.mean(abs(u))
# relative_var = avg * 0.1
# sigma = relative_var 

du1 = np.diff(u, axis=1)
du = np.zeros((n, m))
du[:, :-1] = du1

# sigma = f(u,x)
sigma = 100
# ut_noisy = ut + sigma * np.random.randn(*ut.shape) / np.sqrt(dt)
# ut_no_noisy = ut
ut_noisy = du + sigma * np.random.randn(*du.shape) * np.sqrt(dt)
ut_no_noisy = du
ut = ut_noisy
#print("dt = ", dt)


#Plot
global_min = min(ut_noisy.min(), ut_no_noisy.min())
global_max = max(ut_noisy.max(), ut_no_noisy.max())

# Plot the original (noise-free) ut
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ut_no_noisy, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
plt.colorbar(label='Value')
plt.title('Original (Noise-Free) ut')
plt.xlabel('Time Dimension')
plt.ylabel('Spatial Dimension')

# Plot the noisy ut
plt.subplot(1, 2, 2)
plt.imshow(ut, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
plt.colorbar(label='Value')
plt.title('Noisy ut')
plt.xlabel('Time Dimension')
plt.ylabel('Spatial Dimension')

# Show the plots
plt.tight_layout()
plt.show()


# ### FFT Denoise ###
# #Transform ut_noisy into Fourier space
# #ut_noisy_fft = np.fft.fft2(ut_noisy)  # 2D Fourier Transform
# #ut_no_noisy_fft = np.fft.fft2(ut_no_noisy)

# # Apply a 2D Hanning window
# window = np.outer(windows.hann(ut_noisy.shape[0]), windows.hann(ut_noisy.shape[1]))
# ut_noisy_windowed = ut_noisy * window
# ut_no_noisy_windowed = ut_no_noisy * window

# # remove_window_noisy = ut_noisy_windowed / window
# # remove_window_no_noisy = ut_no_noisy_windowed / window

# # #Test Window Plot
# # plt.figure(figsize=(12, 6))
# # plt.subplot(1, 4, 1)
# # plt.imshow(ut_no_noisy_windowed, aspect='auto', cmap='viridis')
# # plt.colorbar(label='Value')
# # plt.title('Original (Noise-Free) ut Windowed')
# # plt.xlabel('Time Dimension')
# # plt.ylabel('Spatial Dimension')

# # plt.subplot(1, 4, 2)
# # plt.imshow(ut_noisy_windowed, aspect='auto', cmap='viridis')
# # plt.colorbar(label='Value')
# # plt.title('Noisy ut Windowed')
# # plt.xlabel('Time Dimension')
# # plt.ylabel('Spatial Dimension')

# # plt.subplot(1, 4, 3)
# # plt.imshow(remove_window_no_noisy, aspect='auto', cmap='viridis')
# # plt.colorbar(label='Value')
# # plt.title('Original (Noise-Free) ut remove window')
# # plt.xlabel('Time Dimension')
# # plt.ylabel('Spatial Dimension')

# # plt.subplot(1, 4, 4)
# # plt.imshow(remove_window_noisy, aspect='auto', cmap='viridis')
# # plt.colorbar(label='Value')
# # plt.title('Noisy ut remove window')
# # plt.xlabel('Time Dimension')
# # plt.ylabel('Spatial Dimension')

# # plt.tight_layout()
# # plt.show()



# # Perform FFT on windowed data
# ut_noisy_fft = np.fft.fft2(ut_noisy_windowed)
# ut_no_noisy_fft = np.fft.fft2(ut_no_noisy_windowed)

# #Show Fourier Space
# magnitude_spectrum = np.abs(ut_noisy_fft)
# plt.figure(figsize=(8, 6))
# plt.imshow(np.log1p(magnitude_spectrum), aspect='auto', cmap='viridis')
# plt.colorbar(label='Log Magnitude')
# plt.title('Fourier Transform of ut_noisy')
# plt.xlabel('Frequency Dimension 1')
# plt.ylabel('Frequency Dimension 2')

# magnitude_spectrum2 = np.abs(ut_no_noisy_fft)
# plt.figure(figsize=(8, 6))
# plt.imshow(np.log1p(magnitude_spectrum2), aspect='auto', cmap='viridis')
# plt.colorbar(label='Log Magnitude')
# plt.title('Fourier Transform of ut_no_noisy')
# plt.xlabel('Frequency Dimension 1')
# plt.ylabel('Frequency Dimension 2')


# # Step 2: Apply a low-pass filter
# rows, cols = ut_noisy.shape
# crow, ccol = rows // 2, cols // 2  # Center coordinates for reference

# # Create a mask for a low-pass filter (no shift)
# radius = min(rows, cols) // 1  # Define cutoff radius (adjustable)
# mask = np.zeros((rows, cols), dtype=np.float32)
# for i in range(rows):
#     for j in range(cols):
#         # Calculate distance from the top-left corner (default zero-frequency location)
#         distance = np.sqrt(i**2 + j**2)
#         if distance <= radius:
#             mask[i, j] = 1

# # Apply the mask to the Fourier-transformed data
# filtered_fft = ut_noisy_fft * mask

# # Step 3: Transform back to spatial domain
# ut_denoised = np.fft.ifft2(filtered_fft)  # Inverse FFT
# ut_denoised = np.real(ut_denoised)  # Take the real part (discard imaginary noise)

# # Step 4: Remove Window Effect (Compensate for Windowing)
# #ut_denoised = ut_denoised / window

# # Handle division by zero or very small values in the window
# #ut_denoised[np.isinf(ut_denoised)] = 0  # Replace infinities with 0
# #ut_denoised[np.isnan(ut_denoised)] = 0  # Replace NaNs with 0

# # Visualization
# plt.figure(figsize=(12, 8))

# # Original Noisy Data
# plt.subplot(1, 3, 1)
# plt.imshow(ut_noisy, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
# plt.colorbar(label='Value')
# plt.title('Original Noisy Data')

# # Magnitude Spectrum After Filtering (Optional Visualization)
# plt.subplot(1, 3, 2)
# magnitude_spectrum = np.log1p(np.abs(filtered_fft))
# plt.imshow(magnitude_spectrum, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
# plt.colorbar(label='Log Magnitude')
# plt.title('Filtered Magnitude Spectrum')

# # Denoised Data
# plt.subplot(1, 3, 3)
# plt.imshow(ut_denoised, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
# plt.colorbar(label='Value')
# plt.title('Denoised Data')

# plt.tight_layout()
# plt.show()


# calculate error
#Burgers:
right_side = -u*ux+0.1*uxx
left_side = ut
right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin
left_side_origin = ut_origin
#KdV
# right_side = -0.0025 * uxxx - u * ux
# left_side = ut
# right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin
# left_side_origin = ut_origin
#Chafee-Infante 
# right_side = - 1.0008*u + 1.0004*u**3
# left_side = ut
# right_side_origin = uxx_origin-u_origin+u_origin**3
# right_side_origin = 1.0002*uxx_origin-1.0008*u_origin+1.0004*u_origin**3
# left_side_origin = ut_origin
#PDE divide
# right_side = -config.divide(ux, x) + 0.25*uxx
# left_side = ut
# right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin
# left_side_origin = ut_origin
#PDE compound
# right_side = u*uxx + ux*ux
# left_side = ut
# right_side_origin = u_origin*uxx_origin + ux_origin*ux_origin
# left_side_origin = ut_origin

n1, n2, m1, m2 = int(n*0.1), int(n*0.9), int(m*0), int(m*1)
right_side_full = right_side
right_side_1 = right_side[n1:n2, m1:m2]
left_side_1 = left_side[n1:n2, m1:m2]
right = np.reshape(right_side_1, ((n2-n1)*(m2-m1), 1))
left = np.reshape(left_side_1, ((n2-n1)*(m2-m1), 1))
diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1))
print('data error without edges',diff)

n1_origin, n2_origin, m1_origin, m2_origin = int(n_origin*0.1), int(n_origin*0.9), int(m_origin*0), int(m_origin*1)
right_side_full_origin = right_side_origin
right_side_origin_1 = right_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
left_side_origin_1 = left_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
right_origin = np.reshape(right_side_origin_1, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
left_origin = np.reshape(left_side_origin_1, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((n2_origin-n1_origin)*(m2_origin-m1_origin))
print('data error_origin without edges',diff_origin)

# exec (config.right_side)
# exec (config.left_side)
n1, n2, m1, m2 = int(n*0), int(n*1), int(m*0), int(m*1)
right_side_full = right_side
right_side_2 = right_side[n1:n2, m1:m2]
left_side_2 = left_side[n1:n2, m1:m2]
right = np.reshape(right_side_2, ((n2-n1)*(m2-m1), 1))
left = np.reshape(left_side_2, ((n2-n1)*(m2-m1), 1))
diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1))
print('data error',diff)

# exec (config.right_side_origin)
# exec (config.left_side_origin)
n1_origin, n2_origin, m1_origin, m2_origin = int(n_origin*0), int(n_origin*1), int(m_origin*0), int(m_origin*1)
right_side_full_origin = right_side_origin
right_side_origin_2 = right_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
left_side_origin_2 = left_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
right_origin = np.reshape(right_side_origin_2, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
left_origin = np.reshape(left_side_origin_2, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((n2_origin-n1_origin)*(m2_origin-m1_origin))
print('data error_origin',diff_origin)


###########################################################################################
# for default evaluation
# transforming u into a 2D array with: n*m rows and 1 column
default_u = np.reshape(u, (u.shape[0]*u.shape[1], 1))
default_ux = np.reshape(ux, (u.shape[0]*u.shape[1], 1))
default_uxx = np.reshape(uxx, (u.shape[0]*u.shape[1], 1))
# default_uxxx = np.reshape(uxxx, (u.shape[0]*u.shape[1], 1))
default_u2 = np.reshape(u**2, (u.shape[0]*u.shape[1], 1))
default_u3 = np.reshape(u**3, (u.shape[0]*u.shape[1], 1))
# default_terms = np.hstack((default_u, default_ux, default_uxx, default_u2, default_u3))
# default_names = ['u', 'ux', 'uxx', 'u^2', 'u^3']
# default_terms = np.hstack((default_u, default_ux))
# default_names = ['u', 'ux']
default_terms = np.hstack((default_u)).reshape(-1,1)
default_names = ['u']
print(default_terms.shape)
num_default = default_terms.shape[1]

zeros = np.zeros(u.shape)

if simple_mode:
    ALL = [['+', 2, np.add], ['-', 2, np.subtract],['*', 2, np.multiply], 
           ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2], ['u', 0, u], 
           ['x', 0, x], ['ux', 0, ux], ['0', 0, zeros], ['^2', 1, np.square], ['^3', 1, cubic]] ##  ['u^2', 0, u**2], ['uxx', 0, uxx], ['t', 0, t],
    OPS = [['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide],
           ['d', 2, Diff], ['d^2', 2, Diff2], ['^2', 1, np.square], ['^3', 1, cubic]]
    ROOT = [['*', 2, np.multiply], ['d', 2, Diff], ['d^2', 2, Diff2], ['/', 2, divide], ['^2', 1, np.square], ['^3', 1, cubic]]
    OP1 = [['^2', 1, np.square], ['^3', 1, cubic]]
    OP2 = [['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2]]
    # VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux], ['uxx', 0, uxx], ['u^2', 0, u**2]]) 
    VARS = [['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux]] 
    den = [['x', 0, x]]

pde_lib, err_lib = [], []
