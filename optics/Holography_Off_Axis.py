from tqdm import tqdm
import numpy as np
import utils

def get_reference_shifts(first_background_hologram, cutoff):
    N = first_background_hologram.shape[0]
    ref_fourier = np.fft.fftshift(np.fft.fft2(first_background_hologram))
    ref_fourier[:, int(N*cutoff):-1] = 0
    reference_center = utils.get_maxindex(ref_fourier)
    shifts = N//2 - np.array(reference_center)
    return shifts

class Holography_Off_Axis:
    @staticmethod
    def get_object_field(background_hologram, sample_hologram):
        # Calculate reference shifts to retrieve position of F[UR*]
        cutoff = 1/3
        ref_shift = get_reference_shifts(first_background_hologram=background_hologram[0], cutoff=cutoff)

        # Initialize object fields
        Z, N, _ = background_hologram.shape
        background_object_field = np.zeros((Z, N, N), dtype=np.complex64)
        sample_object_field = np.zeros((Z, N, N), dtype=np.complex64)
        
        # Custom filter. Just for separate off-axis term from DC and other off-axis term.
        filter = utils.circular_filter((N, N), pixel_radius=int(N*cutoff/2))
        
        # Iteration for all illumination angle : range(Z)
        for i in tqdm(range(Z)):
            # Normalization
            # normalization = np.sum(background_hologram[i]) / N / N
            # background_hologram_temp = background_hologram[i] / normalization
            # sample_hologram_temp = sample_hologram[i] / normalization
            
            background_hologram_temp = background_hologram[i]
            sample_hologram_temp = sample_hologram[i]
            
            # Fourier transform
            back_temp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(background_hologram_temp)))
            sample_temp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sample_hologram_temp)))
            
            # F[UR*] -> F[U] : Shifting in the fourier domain is equivalent to dividing in the real domain.
            # Then *filter : Remove DC term and another off-axis term
            back_temp = np.roll(back_temp, shift=ref_shift[0], axis=0)
            back_temp = np.roll(back_temp, shift=ref_shift[1], axis=1)
            back_temp = back_temp * filter
            
            sample_temp = np.roll(sample_temp, shift=ref_shift[0], axis=0)
            sample_temp = np.roll(sample_temp, shift=ref_shift[1], axis=1)
            sample_temp = sample_temp * filter
            
            background_object_field[i] = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(back_temp)))
            sample_object_field[i] = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(sample_temp)))
            
            normalization_factor = np.sum(np.abs(background_object_field[i]))/N/N
            
            background_object_field[i] = background_object_field[i] / normalization_factor
            sample_object_field[i] = sample_object_field[i] / normalization_factor
        
        return background_object_field, sample_object_field
        