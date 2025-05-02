import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos
from scipy.special import hankel2 as H2
from numpy.linalg import solve
from scipy import constants

# Setup
f = 300e6  #remove this if we always calculate in wavelenghs???
wavelength = constants.c / f
k = 2 * np.pi / wavelength
eta_0 = sqrt(constants.mu_0 / constants.epsilon_0)
print(eta_0)

# Geometry
a_lambda = 0.5
a = a_lambda * wavelength
N = 100   #Test points (on the boundary)
dphi = 2 * np.pi / N

#Which angle is the incident field coming from? 0=travels in positive x direction
theta_inc_angle = 0
#Radius of observation circle
R_obs_lambda = 3.0
#Amount of observation points
N_obs = 360



# Boundary points
phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
x = a * np.cos(phi)
y = a * np.sin(phi)

#Incident wave
theta_inc = np.radians(theta_inc_angle)
k_vec = k * np.array([np.cos(theta_inc), np.sin(theta_inc)])


def Green2D(x_obs, y_obs, x_src, y_src, k):
    """2D Green's function: scalar, TMz polarization."""
    R = np.sqrt((x_obs - x_src)**2 + (y_obs - y_src)**2)
    return (1j / 4) * H2(0, k * R)

def incident_field(x, y, k_vec):
    """Plane wave incident field: E_inc(x, y) = exp(-j * k • r)"""
    return np.exp(-1j * (k_vec[0] * x + k_vec[1] * y))

def diagonal_term(k, a, dphi):
    """
    Compute the diagonal term for MoM impedance matrix (m = n) with singularity correction.
    This is used for the self-interaction (diagonal elements) in the impedance matrix.
    
    Parameters:
    - k: Wavenumber
    - a: Cylinder radius
    - dphi: Angular segment size (dphi = 2*pi/N)
    
    Returns:
    - The diagonal element for MoM impedance matrix
    """
    return (1j / 4) * (2 * dphi * a) * (1 - 1j * 2 / np.pi * np.log(np.e * k * dphi * a / 2))

# Incident field on boundary
E_inc = incident_field(x, y, k_vec)
#At the surface E=0 => Einc+Esca=0 => Esca = -Einc
V = -E_inc.copy()  # EFIE: E_inc + E_scat = 0

# Construct MoM matrix
Z = np.zeros((N, N), dtype=complex)
for m in range(N):
    for n in range(N):
        if m == n:
            Z[m, n] = diagonal_term(k, a, dphi)  # Diagonal term for m = n
        else:
            Z[m, n] = Green2D(x[m], y[m], x[n], y[n], k) * a * dphi

# Solve the linear equations
Jz = solve(Z, V)

# Observation points
R_obs = R_obs_lambda * wavelength
phi_obs = np.linspace(0, 2 * np.pi, N_obs, endpoint=False)
x_obs = R_obs * np.cos(phi_obs)
y_obs = R_obs * np.sin(phi_obs)

# E_sca at observation
E_scat_obs = np.zeros(N_obs, dtype=complex)
for n in range(N):
    G = Green2D(x_obs, y_obs, x[n], y[n], k)
    E_scat_obs += Jz[n] * G * a * dphi

# Total Field
E_inc_obs = incident_field(x_obs, y_obs, k_vec)
E_total_obs = E_inc_obs + E_scat_obs

# Plot Field Magnitude at observation
plt.figure(figsize=(8, 4))
plt.plot(np.degrees(phi_obs), 20 * np.log10(np.abs(E_total_obs)))
plt.title(f"Total Field on Observation Circle (R = {R_obs_lambda} λ)")
plt.xlabel("Observation Angle (degrees)")
plt.ylabel("Magnitude |E| (dB)")
plt.xticks(np.arange(0, 361, 30))  # Set x-ticks from 0deg to 360deg in steps of 30deg
plt.grid(True)
plt.tight_layout()
plt.show()
