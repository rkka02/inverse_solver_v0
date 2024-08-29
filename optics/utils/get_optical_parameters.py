import numpy as np

def get_optical_parameters(image_shape, lam, n_medium, dx_cam, dx_ol):
    N = image_shape[1]
    parameters = dict()
    
    # Camera parameters
    parameters['dx_cam'] = dx_cam
    parameters['B_cam'] = 1 / 2 / dx_cam
    parameters['dv_cam'] = 2 * parameters['B_cam'] / N
    
    # Objective lens parameters
    parameters['dx_ol'] = dx_ol
    parameters['B_ol'] = 1 / 2 / dx_ol
    parameters['dv_ol'] = 2 * parameters['B_ol'] / N
    
    # Light parameters
    parameters['v0'] = 1 / lam
    parameters['v_nm'] = parameters['v0'] * n_medium
    parameters['k_nm'] = 2 * np.pi * parameters['v_nm']
    parameters['s_nm'] = parameters['v_nm'] // parameters['dv_ol']
    
    return parameters