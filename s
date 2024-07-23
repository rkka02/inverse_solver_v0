illumination_center_list = []
object_field = np.zeros(sample.shape, dtype=np.complex64)

############################
cutoff = 1/3

back_holo = background[0]
back_holo_fourier = np.fft.fftshift(np.fft.fft2(back_holo))

# cutoff
back_holo_fourier[:, int(N*cutoff):-1] = 0
# reference center
reference_center = get_maxindex(back_holo_fourier)
############################

for i in tqdm(range(sample.shape[0])):

    # Background
    back_holo = background[i]
    back_holo_fourier = np.fft.fftshift(np.fft.fft2(back_holo))
    # Centering
    back_holo_fourier = np.roll(back_holo_fourier, shift=N//2-reference_center[0], axis=0)
    back_holo_fourier = np.roll(back_holo_fourier, shift=N//2-reference_center[1], axis=1)
    # Crop
    back_holo_fourier[0:int(N*cutoff), :]=0
    back_holo_fourier[-int(N*cutoff):-1, :]=0
    back_holo_fourier[:, 0:int(N*cutoff)]=0
    back_holo_fourier[:, -int(N*cutoff):-1]=0
    ##################
    # Sample
    sample_holo = sample[i]
    sample_holo_fourier = np.fft.fftshift(np.fft.fft2(sample_holo))
    # Centering
    sample_holo_fourier = np.roll(sample_holo_fourier, shift=N//2-reference_center[0], axis=0)
    sample_holo_fourier = np.roll(sample_holo_fourier, shift=N//2-reference_center[1], axis=1)
    # Crop
    sample_holo_fourier[0:int(N*cutoff), :]=0
    sample_holo_fourier[-int(N*cutoff):-1, :]=0
    sample_holo_fourier[:, 0:int(N*cutoff)]=0
    sample_holo_fourier[:, -int(N*cutoff):-1]=0

    ##################
    back_holo = np.fft.ifft2(back_holo_fourier)
    sample_holo = np.fft.ifft2(sample_holo_fourier)

    object_amplitude = np.abs(sample_holo)
    object_angle = np.angle(sample_holo/back_holo)
    object_field[i] = object_amplitude * np.exp(1j*object_angle)
    







    
    # get illumination center
    illumination_center = [np.where(sample_holo_fourier==sample_holo_fourier.max())[i][0] for i in range(2)]
    illumination_center_list.append(illumination_center)