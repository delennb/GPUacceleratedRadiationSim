import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import gc
import statistics
import pandas as pd

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # Convert to GB
    return cpu_mem, gpu_mem

def setup_constants(n_levels, device):
    # Constants
    n_streams = 4  # Number of streams (4-stream approximation)
    omega_0 = torch.tensor(1.0, device=device, dtype=torch.float32)  # Single scattering albedo
    tau_max = torch.tensor(1.0, device=device, dtype=torch.float32)  # Maximum optical depth
    F0 = torch.tensor(1.0, device=device, dtype=torch.float32)  # Solar irradiance
    mu_0 = torch.tensor(-1.0, device=device, dtype=torch.float32)  # cosine of polar angle for incident solar light
    a_0 = torch.tensor(1.0, device=device, dtype=torch.float32)
    C_i0 = (omega_0 / 2) * a_0
    delta_tau = tau_max / n_levels

    # Constants for 4-stream approximation
    # Polar angles
    mu_1 = torch.sqrt(torch.tensor((3 / 7) + (2 / 7) * np.sqrt(6 / 5), device=device, dtype=torch.float32))
    mu_2 = torch.sqrt(torch.tensor((3 / 7) - (2 / 7) * np.sqrt(6 / 5), device=device, dtype=torch.float32))
    mu_3 = -torch.sqrt(torch.tensor((3 / 7) - (2 / 7) * np.sqrt(6 / 5), device=device, dtype=torch.float32))
    mu_4 = -torch.sqrt(torch.tensor((3 / 7) + (2 / 7) * np.sqrt(6 / 5), device=device, dtype=torch.float32))
    mu = torch.tensor([mu_1, mu_2, mu_3, mu_4], device=device, dtype=torch.float32)

    # Gaussian-Legendre quadrature coefficients
    a_1 = (18 - np.sqrt(30)) / 36
    a_2 = (18 + np.sqrt(30)) / 36
    a_3 = (18 + np.sqrt(30)) / 36
    a_4 = (18 - np.sqrt(30)) / 36
    a = torch.tensor([a_1, a_2, a_3, a_4], device=device, dtype=torch.float32)

    return n_streams, omega_0, delta_tau, mu_0, F0, mu, a

def build_matrix_A_xy(n_levels, n_streams, omega_0, delta_tau, mu, a, device):
    A_xy = torch.zeros((n_levels*n_streams, n_levels*n_streams), device=device)
    
    for L in range(n_levels):
        for k in range(n_levels):
            if k == L:
                M = 2*delta_tau/3
                D = delta_tau
            elif (k - L) == 1:
                M = delta_tau/6
                D = (delta_tau + 1)/2
            elif (L - k) == 1:
                M = delta_tau/6
                D = (delta_tau - 1)/2
            else:
                M = torch.tensor(0.0, device=device)
                D = torch.tensor(0.0, device=device)

            for i in range(n_streams):
                for j in range(n_streams):
                    x = 4*L + i
                    y = 4*k + j
                    C_ij = (omega_0/2)*a[j]
                    delta_ij = torch.tensor(1.0 if i == j else 0.0, device=device)
                    b = (delta_ij - C_ij)/mu[i]
                    A_xy[x,y] = b*M + delta_ij*D
    
    return A_xy

def build_vector_F_x(n_levels, n_streams, omega_0, F0, mu_0, delta_tau, mu, device):
    F_x = torch.zeros((n_levels*n_streams, 1), device=device)
    
    for x in range(n_levels*n_streams):
        i = x % 4
        L = x // 4
        L = torch.tensor(L, device=device, dtype=torch.float32)
        
        term1 = -(mu_0 * ((mu_0 - delta_tau) * torch.exp(delta_tau / mu_0) - mu_0) * 
                torch.exp(-L * delta_tau / mu_0 - delta_tau / mu_0)) / delta_tau
        term2 = (mu_0 * (mu_0 * torch.exp(delta_tau / mu_0) - mu_0 - delta_tau) * 
                torch.exp(-L * delta_tau / mu_0)) / delta_tau
        F_x[x] = ((omega_0*F0)/(4*torch.pi*mu[i]))*(term1 + term2)
    
    return F_x

def setup_radiative_transfer(n_levels):
    # Setup constants on CPU
    n_streams, omega_0, delta_tau, mu_0, F0, mu, a = setup_constants(n_levels, device='cpu')
    
    # Build matrix A_xy on CPU
    A_xy = build_matrix_A_xy(n_levels, n_streams, omega_0, delta_tau, mu, a, device='cpu')
    
    # Build vector F_x on CPU
    F_x = build_vector_F_x(n_levels, n_streams, omega_0, F0, mu_0, delta_tau, mu, device='cpu')
    
    return A_xy, F_x

class PerformanceMetrics:
    def __init__(self):
        self.times = []
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0
        
    def update_memory(self, cpu_mem, gpu_mem):
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_mem)
        self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_mem)
        
    def add_time(self, time_value):
        self.times.append(time_value)
        
    @property
    def avg_time(self):
        return statistics.mean(self.times)
    
    @property
    def std_time(self):
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

def benchmark_performance(n_runs=5):
    # Test different numbers of levels
    n_levels_list = [10, 50, 100]#, 500]#, 1000]
    cpu_metrics = {n: PerformanceMetrics() for n in n_levels_list}
    gpu_metrics = {n: PerformanceMetrics() for n in n_levels_list}
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # First run CPU benchmarks
    print("\nRunning CPU benchmarks...")
    for n_levels in n_levels_list:
        print(f"\nProcessing {n_levels} levels:")
        # Setup A_xy and F_x on CPU
        A_xy_cpu, F_x_cpu = setup_radiative_transfer(n_levels)
        for run in tqdm(range(n_runs)):
            # Clear memory before each run
            gc.collect()
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            start_time = time.time()
            # Solve the system on CPU
            I_y = torch.linalg.solve(A_xy_cpu, F_x_cpu)
            cpu_time = time.time() - start_time
            
            cpu_mem, _ = get_memory_usage()
            cpu_metrics[n_levels].add_time(cpu_time)
            cpu_metrics[n_levels].update_memory(cpu_mem, 0)
                
        print(f"Average CPU time: {cpu_metrics[n_levels].avg_time:.4f} ± {cpu_metrics[n_levels].std_time:.4f} seconds")
        print(f"Peak CPU memory: {cpu_metrics[n_levels].peak_cpu_memory:.2f} GB")
    
    # Then run GPU benchmarks if available
    if device != 'cpu':
        print("\nRunning GPU benchmarks...")
        for n_levels in n_levels_list:
            print(f"\nProcessing {n_levels} levels:")
            # Setup A_xy and F_x on CPU
            A_xy_cpu, F_x_cpu = setup_radiative_transfer(n_levels)
            # Move A_xy and F_x to GPU
            A_xy_gpu = A_xy_cpu.to(device)
            F_x_gpu = F_x_cpu.to(device)
            for run in tqdm(range(n_runs)):
                # Clear memory before each run
                gc.collect()
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                    
                start_time = time.time()
                # Solve the system on GPU
                I_y = torch.linalg.solve(A_xy_gpu, F_x_gpu)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                        
                gpu_time = time.time() - start_time
                    
                cpu_mem, gpu_mem = get_memory_usage()
                gpu_metrics[n_levels].add_time(gpu_time)
                gpu_metrics[n_levels].update_memory(cpu_mem, gpu_mem)
                
            print(f"Average GPU time: {gpu_metrics[n_levels].avg_time:.4f} ± {gpu_metrics[n_levels].std_time:.4f} seconds")
            print(f"Peak GPU memory: {gpu_metrics[n_levels].peak_gpu_memory:.2f} GB")
            if gpu_metrics[n_levels].avg_time > 0:
                print(f"Speedup: {cpu_metrics[n_levels].avg_time/gpu_metrics[n_levels].avg_time:.2f}x")
            else:
                print("Speedup: N/A")
    
    return n_levels_list, cpu_metrics, gpu_metrics

def plot_performance(n_levels_list, cpu_metrics, gpu_metrics, speedup_filename="speedup_comparison.png", memory_filename="memory_usage_comparison.png"):
    # --- Speedup Plot ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Calculate speedup
    speedup = []
    for n in n_levels_list:
        if gpu_metrics[n].avg_time > 0:  # Avoid division by zero
            speedup.append(cpu_metrics[n].avg_time / gpu_metrics[n].avg_time)
        else:
            speedup.append(None)  # Set None for invalid cases
    
    # Plot speedup
    ax1.plot(n_levels_list, speedup, 'g-o', label='Speedup (CPU Time / GPU Time)')
    
    ax1.set_xlabel('Number of Levels')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup: CPU vs GPU')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Save the speedup plot
    plt.tight_layout()
    plt.savefig(speedup_filename)
    print(f"Speedup plot saved to {speedup_filename}")
    plt.close(fig1)

    # --- Memory Usage Plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Memory usage
    cpu_memory = [metrics.peak_cpu_memory for metrics in cpu_metrics.values()]
    gpu_memory = [metrics.peak_gpu_memory if metrics.peak_gpu_memory > 0 else None for metrics in gpu_metrics.values()]

    ax2.plot(n_levels_list, cpu_memory, 'b-o', label='CPU Memory')
    if any(gpu_memory):
        ax2.plot(n_levels_list, gpu_memory, 'r-o', label='GPU Memory')
    
    ax2.set_xlabel('Number of Levels')
    ax2.set_ylabel('Memory Usage (GB)')
    ax2.set_title('Memory Usage: CPU vs GPU')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xscale('log')

    # Save the memory usage plot
    plt.tight_layout()
    plt.savefig(memory_filename)
    print(f"Memory usage plot saved to {memory_filename}")
    plt.close(fig2)

def save_statistics_to_csv_pandas(n_levels_list, cpu_metrics, gpu_metrics, filename="performance_statistics.csv"):
    data = []
    for n_levels in n_levels_list:
        data.append({
            "Number of Levels": n_levels,
            "CPU Avg Time (s)": cpu_metrics[n_levels].avg_time,
            "CPU Std Time (s)": cpu_metrics[n_levels].std_time,
            "CPU Peak Memory (GB)": cpu_metrics[n_levels].peak_cpu_memory,
            "GPU Avg Time (s)": gpu_metrics[n_levels].avg_time if gpu_metrics[n_levels].times else None,
            "GPU Std Time (s)": gpu_metrics[n_levels].std_time if gpu_metrics[n_levels].times else None,
            "GPU Peak Memory (GB)": gpu_metrics[n_levels].peak_gpu_memory if gpu_metrics[n_levels].peak_gpu_memory else None,
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Statistics saved to {filename}")

if __name__ == "__main__":
    n_levels_list, cpu_metrics, gpu_metrics = benchmark_performance(n_runs=5)
    plot_performance(n_levels_list, cpu_metrics, gpu_metrics)
    save_statistics_to_csv_pandas(n_levels_list, cpu_metrics, gpu_metrics, filename="performance_statistics.csv")
