import pandas as pd
import matplotlib.pyplot as plt

def plot_speedup():
    # Load data from CSV files
    mps_data = pd.read_csv("MPS_data.csv")
    cuda_data = pd.read_csv("cuda_data.csv")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot MPS data
    plt.plot(mps_data["Layer Size"], mps_data["Average Speedup"], marker='o', linestyle='-', color='b', label="MPS (Apple GPU)")

    # Plot CUDA data
    plt.plot(cuda_data["Layer Size"], cuda_data["Average Speedup"], marker='s', linestyle='-', color='r', label="CUDA (NVIDIA GPU)")

    # Labels and title
    plt.xlabel('Number of Tau Layers')
    plt.ylabel('Average Speedup (CPU Time / GPU Time)')
    plt.title('GPU Speedup Comparison: MPS vs. CUDA')
    plt.xscale('log')  # Logarithmic scale for layer size
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_speedup()