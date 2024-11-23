import numpy as np
from scipy.linalg import solve

# Constants
n_streams = 4  # Number of streams (4-stream approximation)
n_levels = 10  # Number of atmospheric levels
omega_0 = 0.9  # Single scattering albedo
tau_max = 1.0  # Maximum optical depth
F0 = 1.0  # Solar irradiance

# Discretization
tau = np.linspace(0, tau_max, n_levels)  # Optical depth grid
mu = np.array([0.5, 0.25, -0.25, -0.5])  # Discrete ordinates for 4 streams
weights = np.array([0.5, 0.5, 0.5, 0.5])  # Quadrature weights (Gaussian-Legendre)

# Scattering phase function (isotropic for simplicity)
def phase_function(mu_i, mu_j):
    return 1.0  # Isotropic scattering

# Matrix setup
D = np.zeros((n_streams * n_levels, n_streams * n_levels))  # Coefficient matrix
F = np.zeros(n_streams * n_levels)  # Source vector
I = np.zeros(n_streams * n_levels)  # Solution vector

# Fill matrices
for l in range(n_levels):
    for i in range(n_streams):
        idx = l * n_streams + i  # Global index
        # Add diagonal term (self-interaction)
        D[idx, idx] = mu[i]
        # Add scattering terms
        for j in range(n_streams):
            D[idx, l * n_streams + j] += omega_0 * weights[j] * phase_function(mu[i], mu[j])
        # Add transport terms for next/previous levels
        if l > 0:  # Interaction with the previous level
            D[idx, (l - 1) * n_streams + i] -= mu[i] / (tau[l] - tau[l - 1])
        if l < n_levels - 1:  # Interaction with the next level
            D[idx, (l + 1) * n_streams + i] += mu[i] / (tau[l + 1] - tau[l])
        # Add source term (direct solar contribution)
        F[idx] = F0 * phase_function(mu[i], 1.0) * np.exp(-tau[l] / abs(mu[i])) * (l == 0)

# Solve the system
I = solve(D, F)

# Reshape solution into levels and streams for easy interpretation
I = I.reshape((n_levels, n_streams))

# Output the solution
print("Radiative Intensities (4-Stream Approximation):")
for l in range(n_levels):
    print(f"Level {l}: {I[l]}")

# Optional: Visualization
import matplotlib.pyplot as plt
for i in range(n_streams):
    plt.plot(tau, I[:, i], label=f"Stream {i+1}")
plt.xlabel("Optical Depth (tau)")
plt.ylabel("Radiative Intensity")
plt.title("Radiative Intensity Across Levels")
plt.legend()
plt.show()
