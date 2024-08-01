import numpy as np

def circular_filter(array_shape, pixel_radius):
    filter = np.zeros(array_shape)
    N = array_shape[0]
    x, y = np.arange(N), np.arange(N)
    X, Y = np.meshgrid(x, y)
    filter[(X-N//2)**2 + (Y-N//2)**2 < pixel_radius**2] = 1
    return filter