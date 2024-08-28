import numpy as np
from tqdm import tqdm
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
        cutoff = 1/3
        shifts = get_reference_shifts(first_background_hologram=background_hologram[0], cutoff=cutoff)
        
        Z, N, _ = background_hologram.shape
        background_object_field = np.zeros((Z, N, N), dtype=np.complex64)
        sample_object_field = np.zeros((Z, N, N), dtype=np.complex64)
        
        filter = utils.circular_filter((N, N), pixel_radius=int(N*cutoff/2))
        
        for i in tqdm(range(Z)):
            background_hologram_temp = background_hologram[i]
            sample_hologram_temp = sample_hologram[i]
            
            back_temp = np.fft.fftshift(np.fft.fft2(background_hologram_temp))
            sample_temp = np.fft.fftshift(np.fft.fft2(sample_hologram_temp))
            
            back_temp = np.roll(back_temp, shift=shifts[0], axis=0)
            back_temp = np.roll(back_temp, shift=shifts[1], axis=1)
            back_temp = back_temp * filter
            
            sample_temp = np.roll(sample_temp, shift=shifts[0], axis=0)
            sample_temp = np.roll(sample_temp, shift=shifts[1], axis=1)
            sample_temp = sample_temp * filter
            
            background_object_field[i] = np.fft.ifft2(back_temp)
            sample_object_field[i] = np.fft.ifft2(sample_temp)
            
            normalization_factor = np.sum(np.abs(background_object_field[i]))/N/N
            
            background_object_field[i] = background_object_field[i] / normalization_factor
            sample_object_field[i] = sample_object_field[i] / normalization_factor
        
        return background_object_field, sample_object_field

def project_onto_ewald_sphere(scattered_field, k0, dk, n_medium):
    """
    Project the Fourier transformed scattered field onto the Ewald sphere.
    
    Parameters:
    scattered_field (np.array): The Fourier transformed scattered field
    k0 (float): Wavenumber of the illumination
    dk (float): Sampling frequency in k-space
    n_medium (float): Refractive index of the medium
    
    Returns:
    np.array: The projected field onto the Ewald sphere
    """
    Nx, Ny, Nz = scattered_field.shape
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1/dk))
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=1/dk))
    kz = np.fft.fftshift(np.fft.fftfreq(Nz, d=1/dk))
    
    KX, KY = np.meshgrid(kx, ky)
    KZ = np.sqrt((n_medium * k0)**2 - KX**2 - KY**2)
    
    # Create a mask for the Ewald sphere
    mask = np.real(KZ) > 0
    
    # Initialize the projected field
    projected_field = np.zeros_like(scattered_field)
    
    for i in range(Nz):
        kz_i = kz[i]
        # Find the indices where KZ is close to kz_i
        indices = np.abs(KZ - kz_i) < dk/2
        indices = indices & mask
        
        # Project the field onto the Ewald sphere
        projected_field[:,:,i][indices] = scattered_field[:,:,i][indices]
    
    return projected_field

# Example usage
if __name__ == "__main__":
    # Assume we have already retrieved the scattered field
    scattered_field = np.fft.fftn(sample_object_field - background_object_field)
    
    # Example parameters (you should adjust these based on your setup)
    wavelength = 632.8e-9  # HeNe laser wavelength
    k0 = 2 * np.pi / wavelength
    pixel_size = 6.5e-6  # Camera pixel size
    magnification = 60  # Microscope magnification
    n_medium = 1.33  # Refractive index of water
    
    dk = 2 * np.pi / (pixel_size * magnification * scattered_field.shape[0])
    
    # Project the scattered field onto the Ewald sphere
    projected_field = project_onto_ewald_sphere(scattered_field, k0, dk, n_medium)
    
    # Now you can use the projected_field for further processing or visualization