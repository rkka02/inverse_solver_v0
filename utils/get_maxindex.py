import numpy as np

def get_maxindex(array):
    coords = np.where(array==array.max())
    row = coords[0][0]
    column = coords[1][0]
    return [row, column]

def get_maxindex_3d(array_3d):
    coords = np.where(array_3d==array_3d.max())
    row = coords[0][0]
    column = coords[1][0]
    z = coords[2][0]
    return [row, column, z]