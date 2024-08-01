import utils
import numpy as np
from tqdm import tqdm
from skimage import restoration

def get_scattered_field(background_object_field, sample_object_field):
    # Initialize 
    Z, N, _ = background_object_field.shape
    retrieved_background = np.zeros((Z, N, N), dtype=np.complex128)
    retrieved_scattered_field = np.zeros((Z, N, N), dtype=np.complex128)
        
    for i in tqdm(range(Z)):
        back_temp = background_object_field[i]
        sample_temp = sample_object_field[i]
            
        # background
        back_amplitude = np.abs(back_temp)
        back_angle = np.angle(back_temp)
        # sample
        sample_amplitude = np.abs(sample_temp)
        sample_angle = np.angle(sample_temp)
            
        # born
        scattered_amplitude = sample_amplitude/back_amplitude
        scattered_angle = back_angle-sample_angle
        scattered_angle = utils.phi_shift(restoration.unwrap_phase(scattered_angle))
            
        retrieved_background[i] = back_amplitude * np.exp(1j*back_angle)
        retrieved_scattered_field[i] = scattered_amplitude * np.exp(1j * scattered_angle) - 1
        
    return retrieved_background, retrieved_scattered_field

class Backward_Solve_Born:
    @staticmethod
    def solve(background_object_field, sample_object_field):
        