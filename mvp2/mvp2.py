import math
import numpy as np
import torch
import random
import time
import pandas as pd

def test_prefix_sum(arr):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return torch.cumsum(arr, dim=0)

def intensity_cpu(taus, I_in):
    start_time_cpu = time.time()
    tau_sum = np.cumsum(taus)
    end_time_cpu = time.time()
    time_cpu = end_time_cpu - start_time_cpu
    I = I_in * np.exp(-tau_sum)
    return I, time_cpu

def intensity_gpu(taus, I_in):
    taus = torch.tensor(taus)
    start_time_gpu = time.time()
    tau_sum = test_prefix_sum(taus)
    end_time_gpu = time.time()
    tau_sum = tau_sum.cpu()
    time_gpu = end_time_gpu - start_time_gpu
    I = I_in * np.exp(-tau_sum.numpy())
    return I, time_gpu

def main():
    print(torch.__version__)
    is_cuda_available = torch.cuda.is_available()
    is_mps_available = torch.backends.mps.is_available()
    device_type = "CUDA" if is_cuda_available else "MPS" if is_mps_available else "CPU"
    print(f"Device used: {device_type}")

    # Given incoming light intensity (W/m^2)
    I_inc = 340

    # Define different sizes for layer_taus
    layer_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 1000000]
    average_speedups = []

    for size in layer_sizes:
        cpu_times = []
        gpu_times = []

        # Run 10 times for each size
        for _ in range(10):
            # Generate random tau layers with bias towards small values
            layer_taus = [random.uniform(0, 1) ** 3 for _ in range(size)]

            # Compute intensity and time for CPU
            _, time_cpu = intensity_cpu(layer_taus, I_inc)
            cpu_times.append(time_cpu)

            # Compute intensity and time for GPU
            _, time_gpu = intensity_gpu(layer_taus, I_inc)
            gpu_times.append(time_gpu)

        # Calculate average times for CPU and GPU
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        
        # Calculate average speedup
        average_speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
        average_speedups.append((size, average_speedup))

        print(f"Size: {size}, Avg CPU Time: {avg_cpu_time:.6f}, Avg GPU Time: {avg_gpu_time:.6f}, Avg Speedup: {average_speedup:.2f}")

    # Determine the filename based on the device type
    filename = "cuda_data.csv" if is_cuda_available else "MPS_data.csv" if is_mps_available else "cpu_data.csv"

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(average_speedups, columns=["Layer Size", "Average Speedup"])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()
