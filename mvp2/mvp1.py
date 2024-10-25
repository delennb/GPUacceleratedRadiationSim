import math
import numpy as np
import torch
import random
import time

def test_prefix_sum(arr):
    # Check if CUDA or MPS (Metal Performance Shaders for Mac M1) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")  # Fallback to CPU

    return torch.cumsum(arr, dim=0)

# Function to implement parallel prefix sum using PyTorch
def parallel_prefix_sum(arr):
    # Check if CUDA or MPS (Metal Performance Shaders for Mac M1) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")  # Fallback to CPU

    # Move the array to the selected device (GPU or CPU)
    arr = arr.to(device)

    # Initialize variables
    last_window_size = 1
    current = arr.clone()
    N = arr.shape[0]  # Length of the input array

    # Perform log2(N) iterations
    for i in range(int(torch.ceil(torch.log2(torch.tensor(N, device=device))))):
        # Create a shifted version of the current array
        next_arr = current.clone()

        # Perform the prefix sum computation for the current window size
        # Shifted additions
        if last_window_size < N:
            next_arr[last_window_size:] += current[:-last_window_size]

        # Update window size by doubling it
        last_window_size *= 2
        current = next_arr

    # Move result back to CPU for easy viewing if needed
    return current

def intensity_cpu(taus, I_in):
    start_time_cpu = time.time()
    tau_sum = np.cumsum(taus)
    end_time_cpu =  time.time()
    time_cpu = end_time_cpu - start_time_cpu
    # print(tau_sum)
    I = np.zeros_like(taus)
    I = I_in * np.exp(-tau_sum)
    return I, time_cpu

def intensity_gpu(taus, I_in):
    taus = torch.tensor(taus)
    start_time_gpu = time.time()
    # tau_sum = parallel_prefix_sum(taus)
    tau_sum = test_prefix_sum(taus)
    end_time_gpu = time.time()
    tau_sum = tau_sum.cpu()
    time_gpu = end_time_gpu - start_time_gpu
    I = np.zeros_like(taus)
    I = I_in * np.exp(-tau_sum)
    return I, time_gpu
    # print('bro')

def main():
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.backends.mps.is_available())
    # Given optical depths of layers
    layer_taus = [random.uniform(0, 1) for _ in range(100)]
    layer_taus = [x ** 3 for x in layer_taus]

    # Given incoming light intensity (W/m^2)
    I_inc = 340

    # Compute intensity for each layer
    I_cpu, time_cpu = intensity_cpu(layer_taus, I_inc)
    print(f"Execution time for CPU version: {time_cpu:.6f} seconds")

    I_gpu, time_gpu = intensity_gpu(layer_taus, I_inc)
    print(f"Execution time for GPU version: {time_gpu:.6f} seconds")

    # print('Intensity (CPU): ' + np.array2string(I_cpu))
    # print('Intensity (GPU): ' + str(I_gpu))


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