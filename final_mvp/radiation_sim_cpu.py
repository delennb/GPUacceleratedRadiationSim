import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from scipy.linalg import solve

# Constants
n_streams = 4  # Number of streams (4-stream approximation)
n_levels = 10  # Number of atmospheric levels
omega_0 = 1.0  # Single scattering albedo
tau_max = 1.0  # Maximum optical depth
F0 = 1.0       # Solar irradiance
mu_0 = -1.0    # cosine of polar angle for incident solar light
a_0 = 1
C_i0 = (omega_0/2)*a_0

delta_tau = tau_max/n_levels

# Constants for 4 stream approximation
# Polar angles
mu_1 = np.sqrt((3/7) + (2/7)*np.sqrt(6/5))
mu_2 = np.sqrt((3/7) - (2/7)*np.sqrt(6/5))
mu_3 = -np.sqrt((3/7) - (2/7)*np.sqrt(6/5))
mu_4 = -np.sqrt((3/7) + (2/7)*np.sqrt(6/5))
mu = (mu_1, mu_2, mu_3, mu_4)
# Gaussian-Legendre quadrature coefficients
a_1 = (18 - np.sqrt(30))/36
a_2 = (18 + np.sqrt(30))/36
a_3 = (18 + np.sqrt(30))/36
a_4 = (18 - np.sqrt(30))/36
a = (a_1, a_2, a_3, a_4)

# Set up global matrix A_xy
A_xy = np.zeros((n_levels*n_streams, n_levels*n_streams))

for L in range(0,n_levels):
    for k in range(0,n_levels):
        M = []
        D = []
        if k == L:
            M = 2*delta_tau/3
            D = delta_tau
        if (k - L) == 1:
            M = delta_tau/6
            D = (delta_tau + 1)/2
        if (L - k) == 1:
            M = delta_tau/6
            D = (delta_tau - 1)/2
        if np.abs(L - k) > 1:
            M = 0
            D = 0
        for i in range(1,n_streams+1):
            for j in range(1,n_streams+1):
                x = 4*L + i-1
                y = 4*k + j-1
                C_ij = (omega_0/2)*a[j-1]
                if i == j:
                    delta_ij = 1
                else:
                    delta_ij = 0
                b = (delta_ij - C_ij)/mu[i-1]
                # if M != 0 and D != 0:
                #     print(M)
                #     print(D)
                #     print(b)
                #     print(delta_ij)
                A_xy[x,y] = b*M + delta_ij*D

# print(A_xy)
# print(np.count_nonzero(A_xy))

# Set up global vector F_x
F_x = np.zeros((n_levels*n_streams,1))

for x in range(0,n_levels*n_streams):
    i = x%4 + 1
    L = np.floor(x/4)
    # This formula is for isotropic scattering (P(mu_i, mu_j) = 1)
    term1 = -(mu_0 * ((mu_0 - delta_tau) * np.exp(delta_tau / mu_0) - mu_0) * np.exp(-L * delta_tau / mu_0 - delta_tau / mu_0)) / delta_tau
    term2 = (mu_0 * (mu_0 * np.exp(delta_tau / mu_0) - mu_0 - delta_tau) * np.exp(-L * delta_tau / mu_0)) / delta_tau
    F_x[x] = ((omega_0*F0)/(4*np.pi*mu[i-1]))*(term1 + term2)
    # F_x[x] = ((omega_0*F0)/(4*np.pi*mu[i-1]))*(-mu_0*((mu_0-delta_tau)*np.exp(delta_tau/mu_0) - mu_0)*np.exp(-(L+1)*delta_tau/mu_0)/delta_tau + mu_0*(mu_0*np.exp(delta_tau/mu_0) - mu_0 - delta_tau)*np.exp(-L*delta_tau/mu_0)/delta_tau)
# print(F_x)
I_y = np.dot(np.linalg.inv(A_xy),F_x)
# print("I_y_1:")
print(I_y[2::4])

vec = np.squeeze(I_y[2::4])
print(vec)
# Create the plot
x_values = np.arange(len(vec))
p = figure(title="Plot of Vector Values vs Index", x_axis_label='Index', y_axis_label='Value')
p.line(x_values, vec, line_width=2, legend_label="Vector Data")
p.circle(x_values, vec, size=8, legend_label="Data Points")

# Show the plot
output_notebook()  # Render in notebook if working in one
show(p)