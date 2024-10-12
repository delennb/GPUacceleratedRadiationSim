import math
import numpy as np
import torch

def intensity_cpu(taus, I_in):
    tau_sum = np.cumsum(taus)
    print(tau_sum)
    I = np.zeros_like(taus)
    I = I_in * np.exp(-tau_sum)
    return I

def intensity_gpu(taus, I_in):
    print('bro')

def main():
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    # Given optical depths of layers
    layer_taus = [0.1, 0.4, 0.2, 0.08, 0.6, 1, 1.2, 0.07]

    # Given incoming light intensity (W/m^2)
    I_inc = 340

    # Compute intensity for each layer
    I_cpu = intensity_cpu(layer_taus, I_inc)
    I_gpu = intensity_gpu(layer_taus, I_inc)


    print('Intensity: ' + np.array2string(I_cpu))


if __name__ == "__main__":
    main()

# Notes
    # Lambert's Law - I = I0 * e^(-2)
    #   z = optical depth
    #   z = (0, inf)
    # CPU Version & GPU Version
    # Many layers - n layers
    # z = integral(dz)
    # Use prefix-sum