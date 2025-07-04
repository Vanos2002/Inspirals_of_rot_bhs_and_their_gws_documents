# SIMULATION OF INSPIRALING COMPACT BINARIES up to 3.5PN order - via Cliﬀord M. Will: Post-Newtonian gravitational radiation and equations of motion via direct integration of the relaxed Einstein equations.
# III. Radiation reaction for binary systems with spinning bodies
# K_SSC = 0.5
# Import of the necessary libraries
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
from scipy.integrate import cumulative_trapezoid
from numba import njit
from scipy.signal import savgol_filter
from numpy.polynomial import Polynomial

plt.rcParams['xtick.labelsize'] = 25    # Globally set the x - labelsize
plt.rcParams['ytick.labelsize'] = 25    # Globally set the y - labelsize
mpl.rcParams['agg.path.chunksize'] = 100000   # For enhanced plot quality

np.set_printoptions(precision=15) # Print numbers to 15 significant digits


# Constants (defined for a geometrized unit system)
G = 1         # Gravitational constant
c = 1         # Speed of light

# Set masses in Solar masses
m1_solar = 1    
m2_solar = 1

M_total_solar = m1_solar + m2_solar    # Total Solar mass of the system


# Black hole masses for the code (=<1 ensured) - (geometrized units)
m1 = m1_solar/M_total_solar  # Mass of black hole 1  
m2 = m2_solar/M_total_solar  # Mass of black hole 2  

# Derived quantities
M = m1 + m2  # Total mass of the system, defined as M=1
mu = m1 * m2 / M  # Reduced mass
nu = mu / M   # Symmetric mass ratio
Mc = (m1 * m2)**(3/5) * (m1 + m2)**(-1/5)  # Chirp mass

rsc = 30 * M  # Initial separation scaling parameter (30 times total sytem mass = 30)


# Initial conditions (ensure non-planar motion)
x1_0 = (m2 / M) * rsc       # Initial x-coordinate of black hole 1
y1_0 = 0                    # Initial y-coordinate of black hole 1
z1_0 = (m2 / M) * rsc / 7   # Initial z-coordinate of black hole 1
x2_0 = -(m1 / M) * rsc      # Initial x-coordinate of black hole 2
y2_0 = 0                    # Initial y-coordinate of black hole 2
z2_0 = -(m1 / M) * rsc / 7  # Initial z-coordinate of black hole 2

# Relative separation of the two black holes
r_sep = np.sqrt((x1_0 - x2_0)**2 + (y1_0 - y2_0)**2 + (z1_0 - z2_0)**2)
v0 = np.sqrt(M / r_sep)           # Newtonian circular velocity (for scaling)
vx1_0 = 0                         # Initial x-velocity of black hole 1
vy1_0 = (m2 / M) * v0             # Initial y-velocity of black hole 1
vz1_0 = 0                         # Initial z-velocity of black hole 1
vx2_0 = 0                         # Initial x-velocity of black hole 2
vy2_0 = -(m1 / M) * v0            # Initial y-velocity of black hole 2
vz2_0 = 0                         # Initial z-velocity of black hole 2

# Spin parameters of individual black holes
chi1 = -1.0
chi2 = 0.5
# Initial spin orientation
vec_spin1 = np.array([2/3, -1/3, -2/3])
vec_spin2 = np.array([-2/7, -3/7, 6/7])
# Ensure naked singularity is not achived
assert np.linalg.norm(vec_spin1) <= 1
assert np.linalg.norm(vec_spin2) <= 1
# Initial spin vectors
Sp1_0 = G * m1 * m1 * vec_spin1 * chi1 / c  # Initial proper spin of black hole 1
Sp2_0 = G * m2 * m2 * vec_spin2 * chi2 / c  # Initial proper spin of black hole 2


# FUNCTIONS DEFINITIONS - we employ numba @njit to boost performance
@njit
def reduced_angular_momentum(rvec, vvec):
    L = np.cross(rvec, vvec)  # Reduced angular momentum vector
    return L

# Newtonian acceleration
@njit
def acceleration_N(r, nvec):
    a_N = - M/ r**2 * nvec  # Newtonian acceleration
    
    return a_N


# 1PN acceleration
@njit
def acceleration_1PN(r, nvec, vvec, v, vr):

    a_1PN = - M / r**2 * (nvec * ((1 + 3 * nu) * v**2 - 2 * (2 + nu) * M / r - 3 * nu * vr**2 / 2) - 2 * (2 - nu) * vr * vvec)
    
    
    return a_1PN

# Spin-orbit acceleration (1.5PN)
@njit
def spin_acceleration(Sp, L, r, nvec, vr, vvec, xi):
    
    a_SO = 1 / r**3 * (3 * nvec / (2 *r) * (np.dot(L, (4*Sp + 3 * xi))) - np.cross(vvec, (4*Sp + 3 * xi)) + 3 * vr * np.cross(nvec, (4*Sp + 3 * xi)) / 2)
    
    return a_SO

# 2.5PN acceleration
@njit
def acceleration_2_5PN(r, nvec, vr, v, vvec):
    
    a_2_5PN = 8 * nu * M**2 / (5 * r**3) * (vr * nvec * (3 * v**2 + 17 * M / (3 * r)) - vvec * (v**2 + 3 * M / r))
    
    return a_2_5PN

# 3.5PN acceleration
@njit
def acceleration_3_5PN(vr, r, v, L, nvec, vvec, Sp, xi):
    
    a_3_5PN = - nu * M / (5 * r**4) * (vr * nvec / r * ((120 * v**2 + 280 * vr**2 + 453 * M / r) * np.dot(L, Sp)
              + (285 * v**2 - 245 * vr**2 + 211 * M / r) * np.dot(L, xi)) + vvec / r * (( 87 * v**2
              - 675 * vr**2 - 901 * M / (3 * r)) * np.dot(L, Sp) + 4 * (6 * v**2 - 75 * vr**2 - 41 * M / r) * np.dot(L, xi))
              - 2 * vr * np.cross(vvec, Sp) / 3 * (48 * v**2 + 15 * vr**2 + 364 * M /r) - vr * np.cross(vvec, xi) / 3 * (375 * v**2 - 195 * vr**2 + 640 * M / r)
              + np.cross(nvec, Sp) / 2 * (31 * v**4 - 260 * v**2 * vr**2 + 245 * vr**4 - 689 * v**2 * M / (3 * r)
              + 537 * vr**2 * M / r + 4 * M**2 / (3 * r**2)) - np.cross(nvec, xi) / 2 * (29 * v**4 - 40 * v**2 * vr**2 - 245 * vr**4
              + 211 * v**2 * M / r - 1019 * vr**2 * M / r - 80 * M**2 / r**2))
    
    return a_3_5PN



def total_angular_momentum(L, Sp1, Sp2):
    
    return mu * L + Sp1 + Sp2      # Note that we multiply the scaled angular momentum by reduced mass


# ISCO radius for Kerr black holes
# (defined on https://duetosymmetry.com/tool/kerr-isco-calculator/)
def KerrISCO(M, chi_eff_kerr, Z1, Z2, sign=1):
    # sign = +1 for retrograde (L·S < 0), -1 for prograde (L·S > 0)
    return M * (3 + Z2 + sign * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)))

# Energy radiated
def energy_radiated(vvec, a_2_5PN, a_3_5PN):
    
    return mu * np.dot(vvec, (a_2_5PN + a_3_5PN))

# DEFINE EQUATIONS OF MOTION
def equations_of_motion():
    # Define the equations of motion
    def derivs(y, t):
        # Extract positions, velocities, and spins from the state vector
        x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, \
        Sp1x, Sp1y, Sp1z, Sp2x, Sp2y, Sp2z = y

        # Reconstruct spin vectors
        Sp1 = np.array([Sp1x, Sp1y, Sp1z])
        Sp2 = np.array([Sp2x, Sp2y, Sp2z])
        Sp = Sp1 + Sp2  # Total proper spin vector

        # Relative position and velocity vectors
        rvec = np.array([x1 - x2, y1 - y2, z1 - z2])
        vvec = np.array([vx1 - vx2, vy1 - vy2, vz1 - vz2])

        eps = 1e-12
        r = max(np.linalg.norm(rvec), eps)
        nvec = rvec / r  # Unit vector
        vr = np.dot(nvec, vvec)  # Radial velocity component
        L = reduced_angular_momentum(rvec, vvec)  # Angular momentum vector
        xi = (m2 * Sp1 / m1) + (m1 * Sp2 / m2)  # Effective proper spin vector
        v = np.linalg.norm(vvec)  # Magnitude of relative velocity vector

        # Calculate accelerations
        a_N = acceleration_N(r, nvec)
        a_1PN = acceleration_1PN(r, nvec, vvec, v, vr)
        a_SO = spin_acceleration(Sp, L, r, nvec, vr, vvec, xi)
        a_2_5PN = acceleration_2_5PN(r, nvec, vr, v, vvec)
        a_3_5PN = acceleration_3_5PN(vr, r, v, L, nvec, vvec, Sp, xi)
        

        # Update velocities and positions
        dvx1dt = m2 / M * (a_N[0] + a_1PN[0] + a_SO[0] + a_2_5PN[0] + a_3_5PN[0])
        dvy1dt = m2 / M * (a_N[1] + a_1PN[1] + a_SO[1] + a_2_5PN[1] + a_3_5PN[1])
        dvz1dt = m2 / M * (a_N[2] + a_1PN[2] + a_SO[2] + a_2_5PN[2] + a_3_5PN[2])

        dvx2dt = - m1 / M * (a_N[0] + a_1PN[0] + a_SO[0] + a_2_5PN[0] + a_3_5PN[0])
        dvy2dt = - m1 / M * (a_N[1] + a_1PN[1] + a_SO[1] + a_2_5PN[1] + a_3_5PN[1])
        dvz2dt = - m1 / M * (a_N[2] + a_1PN[2]+ a_SO[2] + a_2_5PN[2] + a_3_5PN[2])

        # Update spin vectors
        dSp1dt = nu * M / r**3 * np.cross(L, Sp1) * (2 + 3 * m2 / (2 * m1))
        dSp2dt = nu * M / r**3 * np.cross(L, Sp2) * (2 + 3 * m1 / (2 * m2))

        return [vx1, vy1, vz1, dvx1dt, dvy1dt, dvz1dt,
                vx2, vy2, vz2, dvx2dt, dvy2dt, dvz2dt,
                dSp1dt[0], dSp1dt[1], dSp1dt[2], dSp2dt[0], dSp2dt[1], dSp2dt[2]]

    return derivs


# Time parameters (code units)
t_start = 0.0      # Start time
t_end = 1000000    # End time
t_steps = 5000000  # Number of time steps
t = np.linspace(t_start, t_end, t_steps)  # Time array
# Initial conditions vector
y0 = np.array([x1_0, y1_0, z1_0, vx1_0, vy1_0, vz1_0,
                x2_0, y2_0, z2_0, vx2_0, vy2_0, vz2_0,
                Sp1_0[0], Sp1_0[1], Sp1_0[2], Sp2_0[0], Sp2_0[1], Sp2_0[2]])


# Get the original derivs function (which expects y, t)
f_orig = equations_of_motion()

# Wrap it for solve_ivp (which expects t, y)
def f_wrapped(t, y):
    return f_orig(y, t)

# Solve using solve_ivp with tight tolerances (employing an 8th order RK SciPy integrator)
sol = solve_ivp(
    fun=f_wrapped,
    t_span=(t_start, t_end),
    y0=y0,
    t_eval=t,
    method='DOP853',      
    atol=1e-18,
    rtol=2.23e-14
)

# Transpose to match odeint shape: (time, variables)
solution = sol.y.T


# Extract the results
x1_sol = solution[:, 0]
y1_sol = solution[:, 1]
z1_sol = solution[:, 2]
vx1_sol = solution[:, 3]
vy1_sol = solution[:, 4]
vz1_sol = solution[:, 5]
x2_sol = solution[:, 6]
y2_sol = solution[:, 7]
z2_sol = solution[:, 8]
vx2_sol = solution[:, 9]
vy2_sol = solution[:, 10]
vz2_sol = solution[:, 11]
Sp1x_sol = solution[:, 12]
Sp1y_sol = solution[:, 13]
Sp1z_sol = solution[:, 14]
Sp2x_sol = solution[:, 15]
Sp2y_sol = solution[:, 16]
Sp2z_sol = solution[:, 17]
# Plotting the results

r_vec = np.array([x1_sol - x2_sol, y1_sol - y2_sol, z1_sol - z2_sol])
v_vec = np.array([vx1_sol - vx2_sol, vy1_sol - vy2_sol, vz1_sol - vz2_sol])
r_sol = np.linalg.norm(r_vec, axis=0) 
theta_t = np.arccos((z1_sol - z2_sol) / np.sqrt((x1_sol - x2_sol)**2 + (y1_sol - y2_sol)**2 + (z1_sol - z2_sol)**2))
phi_t = np.arctan2(y1_sol-y2_sol, x1_sol-x2_sol)  # Angle in the xy-plane

# Ensure phi is unwrapped (to avoid 2π jumps)
phi_unwrapped = np.unwrap(phi_t)

theta_unwrapped = np.unwrap(theta_t)

theta_smooth = gaussian_filter1d(theta_unwrapped, sigma=30)

phi_smooth = gaussian_filter1d(phi_unwrapped, sigma=80)


# Convert code time units to real seconds (physical SI units)
M_sun = 1.98847e30  # kg
G_SI = 6.6743e-11   # m^3/kg/s^2
c_SI = 299792458    # m/s

M_kg = M_total_solar * M_sun                 # Total system mass in kg)
time_unit_seconds = G_SI * M_kg / c_SI**3    # 1 code unit in seconds
sep_unit_meters = G_SI * M_kg / c_SI**2      # 1 code unit in meters
t_sec = t * time_unit_seconds                # Code time in seconds

# Choose window_length (odd, e.g. 101) and polyorder (e.g. 3)
window_length = 101  # Must be odd and < len(phi_smooth)
polyorder = 3

# Smooth phase (optional, but can help)
phi_smooth_sg = savgol_filter(phi_unwrapped, window_length, polyorder)

# Compute derivative (1st derivative)
dphi_dt_sg = savgol_filter(phi_unwrapped, window_length, polyorder, deriv=1, delta=np.mean(np.diff(t_sec)))
# Post-processing: Trim data at merger
merger_occurred = False
i_final = len(t) 

for i in range(1, len(t)):
    dist = np.sqrt((x1_sol[i] - x2_sol[i])**2 +
                   (y1_sol[i] - y2_sol[i])**2 +
                   (z1_sol[i] - z2_sol[i])**2)
    
    rvec = np.array([x1_sol[i] - x2_sol[i], y1_sol[i] - y2_sol[i], z1_sol[i] - z2_sol[i]])
    vvec = np.array([vx1_sol[i] - vx2_sol[i], vy1_sol[i] - vy2_sol[i], vz1_sol[i] - vz2_sol[i]])
    L = reduced_angular_momentum(rvec, vvec)
    L_nonred = mu * np.linalg.norm(L)
    
    Sp1_sol = np.array([Sp1x_sol[i], Sp1y_sol[i], Sp1z_sol[i]])  # Proper spin of black hole 1
    Sp2_sol = np.array([Sp2x_sol[i], Sp2y_sol[i], Sp2z_sol[i]])  # Proper spin of black hole 2
    Sp_sol = Sp1_sol + Sp2_sol  # Total proper spin vector
    J = total_angular_momentum(L, Sp1_sol, Sp2_sol)  # Total angular momentum

    # Define normalized vectors (ensure nondiverging field)
    L_hat = L / (np.linalg.norm(L) + 1e-14)
    S1_hat = Sp1_sol / (np.linalg.norm(Sp1_sol) + 1e-14)
    S2_hat = Sp2_sol / (np.linalg.norm(Sp2_sol) + 1e-14)

    S_tot = Sp_sol
    chi_eff_kerr = np.dot(S_tot, L_hat) / M**2
    chi_eff_kerr = np.clip(chi_eff_kerr, -1.0, 1.0)  # Enforce physical bound

    # Determine ISCO direction: prograde (sign=-1), retrograde (sign=+1)
    LS_dot = np.dot(L, S_tot)
    sign = -1 if LS_dot > 0 else +1  # -1 for prograde, +1 for retrograde

    # Kerr ISCO auxiliary quantities
    Z1 = 1 + (1 - chi_eff_kerr**2)**(1/3) * (
            (1 + chi_eff_kerr)**(1/3) + (1 - chi_eff_kerr)**(1/3))
    Z2 = np.sqrt(3 * chi_eff_kerr**2 + Z1**2)

    # ISCO radius
    r_ISCO = KerrISCO(M, chi_eff_kerr, Z1, Z2, sign=sign)

    # Stop integration at ISCO (total separation ought to be greater than 2 * r_ISCO, else stop simulation)
    if dist < 2 * r_ISCO:
        merger_occurred = True
        i_final = i
        print(f"Merger detected at t = {t[i]:.4f}, r_ISCO (M) = {r_ISCO:.3f}")
        print(f"Merger detected (real time): {t[i] * time_unit_seconds:.4f} s")
        print(f"KERR ISCO radius (meters): r_ISCO ={r_ISCO * sep_unit_meters:.3f}")
        break
    


# Stops arrays if merger happened
if merger_occurred:
    x1_sol = x1_sol[:i_final]
    y1_sol = y1_sol[:i_final]
    z1_sol = z1_sol[:i_final]
    x2_sol = x2_sol[:i_final]
    y2_sol = y2_sol[:i_final]
    z2_sol = z2_sol[:i_final]
    t = t[:i_final]
    r_sol = r_sol[:i_final]
    phi_t = phi_t[:i_final]
    phi_smooth = phi_smooth[:i_final]
    dphi_dt_sg = dphi_dt_sg[:i_final]
    theta_t = theta_t[:i_final]
    theta_smooth = theta_smooth[:i_final]
    r_vec = r_vec[:, :i_final]
    v_vec = v_vec[:, :i_final]
    Sp1x_sol = Sp1x_sol[:i_final]
    Sp1y_sol = Sp1y_sol[:i_final]
    Sp1z_sol = Sp1z_sol[:i_final]
    Sp2x_sol = Sp2x_sol[:i_final]
    Sp2y_sol = Sp2y_sol[:i_final]
    Sp2z_sol = Sp2z_sol[:i_final]



# Compute time to coalescence array (ensures one plots the graphs with 0 on the right side)
t_to_coalescence = (t[-1] - t) * time_unit_seconds

phi_t_degrees = np.degrees(phi_t)                 # Convert phi_t to degrees
theta_t_degrees= np.degrees(theta_t)              # Convert theta_t to degrees
phi_smooth_degrees = np.degrees(phi_smooth)       # Smoothen phi
theta_smooth_degrees = np.degrees(theta_smooth)   # Smoothen theta



# Compute GW frequency (smoothen)
f_gw_numeric = dphi_dt_sg / np.pi
f_gw_smooth = gaussian_filter1d(f_gw_numeric, sigma=100)
#f_gw_smooth_sg = savgol_filter(f_gw_numeric, window_length=101, polyorder=5)

# ENERGY FLUX CALCULATION
dE_dt_list = []
for i in range(len(t)):
    rvec_i = np.array([x1_sol[i] - x2_sol[i],
                       y1_sol[i] - y2_sol[i],
                       z1_sol[i] - z2_sol[i]])
    vvec_i = np.array([vx1_sol[i] - vx2_sol[i],
                       vy1_sol[i] - vy2_sol[i],
                       vz1_sol[i] - vz2_sol[i]])
    r = np.linalg.norm(rvec_i)
    nvec = rvec_i / r
    vr = np.dot(nvec, vvec_i)
    v = np.linalg.norm(vvec_i)
    L = np.cross(rvec_i, vvec_i)
    Sp1 = np.array([Sp1x_sol[i], Sp1y_sol[i], Sp1z_sol[i]])
    Sp2 = np.array([Sp2x_sol[i], Sp2y_sol[i], Sp2z_sol[i]])
    Sp = Sp1 + Sp2
    xi = (m2 * Sp1 / m1) + (m1 * Sp2 / m2)

    # Radiation-reaction terms
    a_2_5 = acceleration_2_5PN(r, nvec, vr, v, vvec_i)
    a_3_5 = acceleration_3_5PN(vr, r, v, L, nvec, vvec_i, Sp, xi)

    dE_dt = energy_radiated(vvec_i, a_2_5, a_3_5)
    dE_dt_list.append(dE_dt)

dE_dt_array = np.array(dE_dt_list)

# Truncate t_to_coalescence if needed
min_len = min(len(t_to_coalescence), len(dE_dt_array))
t_trimmed = t_to_coalescence[:min_len]
dE_dt_trimmed = dE_dt_array[:min_len]

t_sec_trimmed = t_sec[:min_len]                # Time in seconds

# Total energy radiated (in code units or SI, depending on units of dE_dt)
E_radiated_total = cumulative_trapezoid(dE_dt_array, t_sec_trimmed, initial=0.0)

#CONVERT TO PHYSICAL UNITS - NOTE THE DIVISION BY time_unit_seconds, which was before integrated with
E_totalrad_physical = E_radiated_total * M_kg * c_SI**2 / time_unit_seconds

E_system_total = M_kg * c_SI**2

print(f"Initial Black hole seperation (in km) = {r_sep * G_SI * M_kg / (1000 * c_SI**2):.4f} (km)")
print(f"Total radiated energy: {E_totalrad_physical[-1]/ 1e44:.5f} (*1e51 ergs)")
print(f"Radiated power: {E_totalrad_physical[-1]/(t[-1] * time_unit_seconds) / 1e44:.5f} (*1e51 erg/s)")
print(f"Percentage of energy radiated: {100 * np.abs(E_totalrad_physical[-1])/ E_system_total :.5f} %")


def plot_phi_t(phi_smooth, t_to_coalescence):
    plt.figure(figsize=(10, 6))
    plt.plot(t_to_coalescence, phi_smooth, label=r'$\phi$ vs $t$', color='black', linewidth=3.0)
    plt.xlim(0, t_to_coalescence[0])
    plt.ylim(bottom=0)
    plt.xlabel(r'$\tau$ (s)', fontsize=30)
    plt.ylabel(r'$\phi$ (rad)', fontsize=30)
    plt.grid()
    plt.gca().invert_xaxis()  # So that 0 is on the right
    plt.show()

def plot_theta_t(theta_smooth_degrees, t_to_coalescence):
    plt.figure(figsize=(10, 6))
    plt.plot(t_to_coalescence, theta_smooth_degrees, label=r'$\theta$ vs $t$', color='black')
    plt.xlim(0, t_to_coalescence[0])
    plt.xlabel(r'$\tau$ (s)', fontsize=30)
    plt.ylabel(r'$\theta\ (^{\circ})$', fontsize=30)
    plt.grid()
    plt.gca().invert_xaxis()  # So that 0 is on the right
    # Set x-axis ticks at multiples of 4*pi
    plt.show()

def plot_r(r_sol, t_to_coalescence):
    plt.figure(figsize=(10, 6))
    plt.plot(t_to_coalescence, r_sol, label=r'$r$ vs $t$', color='black')
    plt.xlim(0, t_to_coalescence[0])
    plt.xlabel(r'$\tau$ (s)', fontsize=30)
    plt.ylabel(r'$r$ (M)', fontsize=30)
    plt.grid()
    plt.gca().invert_xaxis()  # So that 0 is on the right
    plt.show()


# Now do all plotting here, after trimming
plot_phi_t(phi_smooth, t_to_coalescence)
plot_theta_t(theta_smooth_degrees, t_to_coalescence)
plot_r(r_sol, t_to_coalescence)


# Compute relative velocity magnitude
v_rel = np.linalg.norm(v_vec, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(r_sol, v_rel, color='purple', linewidth=2)
plt.xlabel(r'Separation $r$ (M)', fontsize=18)
plt.ylabel(r'Relative velocity $|v|$ (code units)', fontsize=18)
plt.title('Relative velocity vs. separation', fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()

# APPROXIMATE THE ORBITAL PHASE USING THE Polynomial.fit
n = 10
p = Polynomial.fit(t_to_coalescence, phi_smooth, deg=n)

# Evaluate polynomial at original points
phi_approx = p(t_to_coalescence)

# Plot (comparison to polynomial plot)
plt.plot(t_to_coalescence, phi_smooth, label='Original orbital phase', alpha=0.5)
plt.plot(t_to_coalescence, phi_approx, label=f'Polynomial fit degree {n}', linewidth=2)
plt.xlim(0, t_to_coalescence[0])
plt.ylim(bottom=0)
plt.xlabel(r'$\tau$ (s)', fontsize=30)
plt.ylabel(r'$\phi (rad)$', fontsize=30)
plt.legend()
plt.gca().invert_xaxis()  # So that 0 is on the right
#plt.show()

p_monomial = p.convert()
coeffs = p_monomial.coef  # These are the a_0, a_1, ..., a_n coefficients

# Print the polynomial approximation of phi
for i, c in enumerate(coeffs):
    print(f"a_{i} = {c:.6e}")

# GRAVITATIONAL WAVE FREQUENCY AT MERGER
print(f"GW frequency (numerically): {f_gw_smooth[-1]:.4f} Hz")

# r_vec and v_vec are (3, N), so transpose to (N, 3)
r_vec_T = r_vec.T  # shape (N, 3)
v_vec_T = v_vec.T  # shape (N, 3)

L_vec = mu * np.cross(r_vec_T, v_vec_T)  # shape (N, 3)
L_norm = np.linalg.norm(L_vec, axis=1)  

# PRECESSION FREQUENCIES
Omega_prec_1_vec = (
    nu * M / (r_sol**3) * (2 + 3 * m2 / (2 * m1))
)[:, np.newaxis] * L_vec  # Precession frequency
Omega_prec_2_vec = (
    nu * M / (r_sol**3) * (2 + 3 * m1 / (2 * m2))
)[:, np.newaxis] * L_vec  # Precession frequency

Omega_prec_1 = np.linalg.norm(Omega_prec_1_vec, axis=1)
Omega_prec_2 = np.linalg.norm(Omega_prec_2_vec, axis=1)

Omega_1_prec_Hz = Omega_prec_1/ (2 * np.pi * time_unit_seconds)       # Convert to Hz
Omega_prec_2_Hz = Omega_prec_2 / (2 * np.pi * time_unit_seconds)      # Convert to Hz

# Precession frequencies
plt.figure(figsize=(10, 6))
plt.plot(t_to_coalescence, Omega_1_prec_Hz, label=r'$\Omega_{1,\mathrm{prec}}$ (Hz)', color='blue', linewidth=2)
plt.plot(t_to_coalescence, Omega_prec_2_Hz, label=r'$\Omega_{2,\mathrm{prec}}$ (Hz)', color='red', linewidth=2)
plt.xlim(0, t_to_coalescence[0])
plt.ylim(bottom=0)
plt.xlabel(r'$\tau$ (s)', fontsize=30)
plt.ylabel(r'$\Omega_{\mathrm{prec}}$ (Hz)', fontsize=30)
plt.title('Orbital Precession Frequencies')
plt.legend(fontsize=18, loc='upper right')
plt.grid(True)
plt.gca().invert_xaxis()  # So that 0 is on the right
#plt.show()

# Plot the numeric derivative obtained GW frequency - unsmoothness may occur
plt.figure(figsize=(5, 5))
plt.plot(t_to_coalescence, f_gw_numeric, label=r'$h_+$', color='black', linewidth=3.0)
plt.xlim(0, t_to_coalescence[0])
plt.ylim(0, 1000)
plt.xlabel(r'$\tau$ (s)', fontsize=30)
plt.ylabel(r'$f_{\mathrm{gw}}$ (Hz)', fontsize=30)
#plt.title('GW Frequency')
#plt.legend(fontsize=30, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.gca().invert_xaxis()  # So that 0 is on the right
#plt.show()

# GRAVITATIONAL WAVE FREQUENCY AS THE DERIVATIVE OF POlYNOMIAL APPROXIMATION OF THE ORBITAL PHASE
coeffs = [
    5.424771e+02,    # a_0
   -1.505225e+03,    # a_1
    6.250051e+03,    # a_2
   -4.057239e+04,    # a_3
    2.022536e+05,    # a_4
   -6.975817e+05,    # a_5
    1.620254e+06,    # a_6
   -2.476666e+06,    # a_7
    2.382102e+06,    # a_8
   -1.304675e+06,    # a_9
    3.099647e+05     # a_10
]

# Create the polynomial and its derivative
p = np.poly1d(coeffs[::-1])  # np.poly1d expects highest degree first
dp = p.deriv()

f_gw = -dp(t_to_coalescence) / np.pi          # The negative sign is because of time inversion

# Plot the gravitational wave frequency
plt.figure(figsize=(8, 5))
plt.plot(t_to_coalescence, f_gw, color='black', linewidth=3)
plt.xlim(0, t_to_coalescence[0])
plt.ylim(bottom=0)
plt.xlabel(r'$\tau$ (s)', fontsize=30)
plt.ylabel(r'$f_{\mathrm{gw}}$ (Hz)', fontsize=30)
#plt.title('GW frequency from polynomial phase fit', fontsize=18)
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
#plt.show()


cos_iota = L_vec[:, 2] / L_norm
cos_iota = np.clip(cos_iota, -1.0, 1.0)
iota = np.arccos(cos_iota)               # INCLINATION ANGLE DEFINITION
# WAVEFORM CALCULATION
Mcr = Mc * M_sun                # Chirp mass in [kg]
MpC = 3.085677581e+22           # 1 Megaparsec in [m]
R_obs = 150 * MpC               # Distance to the observer in [m]
#iota = np.pi/4                 # Inclination angle (45 degrees) - alternative definition (commented out)
amp = (1 / R_obs) * (G_SI * Mcr / c_SI**2)**(5/4) * (5 / (c_SI * t_to_coalescence))**(1/4)        # GW amplitude (dimensionless)


# Waveform polarizations definitions
h_plus = amp * (1 + np.cos(iota)**2) / 2 * np.cos(2 * phi_t)
h_cross = amp * np.cos(iota) * np.sin(2 * phi_t)


# GW Waveform plot for whole tau, later we select concrete time intervals
plt.figure(figsize=(5, 5))
plt.plot(t_to_coalescence, h_plus, label=r'$h_+$', color='darkblue', linewidth=1.0)
plt.plot(t_to_coalescence, h_cross, label=r'$h_\times$', color='red', linewidth=1.0, alpha=0.9)
plt.xlim(0, t_to_coalescence[0])
plt.xlabel(r'$\tau$ (s)', fontsize=18)
plt.ylabel('Strain amplitude', fontsize=18)
plt.title('Leading-order Gravitational Waveform from Inspiral')
plt.legend(fontsize=18, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=15)
plt.gca().invert_xaxis()  # So that 0 is on the right
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectories
ax.plot(x1_sol, y1_sol, z1_sol, label='Black Hole 1', color='blue')
ax.plot(x2_sol, y2_sol, z2_sol, label='Black Hole 2', color='red')

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# SHOW THE INSPIRALLING BINARY SYSTEM
#plt.show()

# Select only the last 0.35 seconds to coalescence
mask = t_to_coalescence <= 0.35  # Last 0.35 seconds
phi_last0_35 = phi_t[mask]

# Unwrap the phase to get a continuous phase evolution
phi_unwrapped = np.unwrap(phi_last0_35)

# Compute the number of cycles: (total phase change) / (2*pi)
n_cycles = (phi_unwrapped[-1] - phi_unwrapped[0]) / (2 * np.pi)

print(f"Number of full orbital phase cycles in the last 0.35 seconds: {n_cycles:.2f}")

plt.figure(figsize=(5, 5))
plt.plot(t_to_coalescence[mask], h_plus[mask], label=r'$h_+$', color='darkblue', linewidth=2.5)
plt.xlim(0.35, 0)
plt.ylim(-0.5e-22, 0.5e-22)
plt.xlabel(r'$\tau$ (s)', fontsize=18)
plt.ylabel('Strain amplitude', fontsize=18)
#plt.title('Last 0.20 seconds: Gravitational Waveform $h_+$')
plt.legend(fontsize=18, loc='upper right')
plt.tick_params(axis='both', labelsize=15)
plt.grid(True)
plt.tight_layout()
#plt.gca().invert_xaxis()  # So that 0 is on the right
plt.show()

# GW WAVEFORM FOR A CONCRETE TIME INTERVAL (DENOTED AS mask)
plt.figure(figsize=(5, 5))
plt.plot(t_to_coalescence[mask], h_plus[mask], label=r'$h_+$', color='darkblue', linewidth=3.0)
plt.plot(t_to_coalescence[mask], h_cross[mask], label=r'$h_\times$', color='red', linewidth=3.0, alpha=0.9)
plt.xlim(0.35, 0)
plt.ylim(-0.5e-22, 0.5e-22)
plt.xlabel(r'$\tau$ (s)', fontsize=18)
plt.ylabel('Strain amplitude', fontsize=18)
#plt.title('Last 0.20 seconds: Gravitational Waveform $h_+$')
plt.legend(fontsize=18, loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=15)
#plt.gca().invert_xaxis()  # So that 0 is on the right
#plt.show()


# Extract spin vectors from solution
S1_vec = np.array([Sp1x_sol, Sp1y_sol, Sp1z_sol]).T  # shape (N, 3)
S2_vec = np.array([Sp2x_sol, Sp2y_sol, Sp2z_sol]).T  # shape (N, 3)
S_vec = S1_vec + S2_vec  # Total spin vector

# Normalize the spin vectors
S1_norm = np.linalg.norm(S1_vec, axis=1)
S2_norm = np.linalg.norm(S2_vec, axis=1)
S_norm = np.linalg.norm(S_vec, axis=1)

#RELATIVE ERROR OF SPIN MAGNITUDES
S1norm_0 = S1_norm[0]
rel_err_S1 = np.abs(S1_norm - S1norm_0) / (np.abs(S1norm_0) )
S2norm_0 = S2_norm[0]
rel_err_S2 = np.abs(S2_norm - S2norm_0) / (np.abs(S2norm_0) )

plt.figure(figsize=(10, 6))
plt.plot((t[-1] - t), rel_err_S1, label=r'$\delta_{||\vec{\mathcal{S}}_1||}$', color='green')
plt.plot((t[-1] - t), rel_err_S2, label=r'$\delta_{||\vec{\mathcal{S}}_2||}$', color='orange')
plt.xlim(0, (t[-1] - t)[0])
plt.ylim(1e-16, 1e-13)
plt.yscale('log')
plt.xlabel(r'$\tau$ (M)', fontsize=40)
plt.ylabel(r'$\delta_{||\vec{\mathcal{S}}_i||}$', fontsize=40)
#plt.title(r'Relative Error of Spin Magnitudes vs Time')
plt.legend(fontsize=40, loc='lower right')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

# Compute cosines of angles between L and S1, L and S2
cos_theta_L_S1 = np.sum(L_vec * S1_vec, axis=1) / (L_norm * S1_norm + 1e-14)
cos_theta_L_S2 = np.sum(L_vec * S2_vec, axis=1) / (L_norm * S2_norm + 1e-14)
cos_theta_L_S = np.sum(L_vec * S_vec, axis=1) / (L_norm * S_norm + 1e-14)
cos_theta_S1_S2 = np.sum(S1_vec * S2_vec, axis=1) / (S1_norm * S2_norm + 1e-14)

theta_L_S1 = np.arccos(cos_theta_L_S1)
theta_L_S2 = np.arccos(cos_theta_L_S2)
theta_L_S = np.arccos(cos_theta_L_S)
theta_S1_S2 = np.arccos(cos_theta_S1_S2)

theta_L_S1_deg = np.degrees(theta_L_S1)
theta_L_S2_deg = np.degrees(theta_L_S2)
theta_L_S_deg = np.degrees(theta_L_S)
theta_S1_S2_deg = np.degrees(theta_S1_S2)



# Plot of THETA_{LS1} and THETA_{LS2} COMPARISON
plt.figure(figsize=(10, 6))
plt.plot(t_to_coalescence, theta_L_S1_deg, label=r'$\theta_{\vec{\mathcal{L}},\vec{\mathcal{S}}_1}$', color='green', linewidth=3)
plt.plot(t_to_coalescence, theta_L_S2_deg, label=r'$\theta_{\vec{\mathcal{L}},\vec{\mathcal{S}}_2}$', color='orange', linewidth=3)
plt.xlim(0, t_to_coalescence[0])
plt.xlabel(r'$\tau$ (s)', fontsize=30)
plt.ylabel(r'$\theta\ (^{\circ})$', fontsize=30)
#plt.title(r'Alignment: $\vec{L}$ with $\vec{S}_1$ and $\vec{S}_2$ vs Time')
plt.legend(fontsize=30, loc='best')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

# Plot of THETA_{LS} and THETA_{S1S2} COMPARISON
plt.figure(figsize=(10, 6))
plt.plot(t_to_coalescence, theta_L_S_deg, label=r'$\theta_{\vec{\mathcal{L}},\vec{\mathcal{S}}}$', color='darkblue')
plt.plot(t_to_coalescence, theta_S1_S2_deg, label=r'$\theta_{\vec{\mathcal{S}}_1,\vec{\mathcal{S}}_2}$', color='red')
plt.xlim(0, t_to_coalescence[0])
plt.xlabel(r'$\tau$ (s)', fontsize=30)
plt.ylabel(r'$\theta\ (^{\circ})$', fontsize=30)
#plt.title(r'Alignment: $\vec{L}$ with $\vec{S}_1$ and $\vec{S}_2$ vs Time')
plt.legend(fontsize=30, loc='best')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()


Xivec_sol = (m2 * S1_vec / m1) + (m1 * S2_vec / m2)     # Effective spin
 
# Plot THE 3D TRAJECTORY OF SPIN AND ANGULAR MOMENTUM (ALTERNATIVELY FOR EFFECTIVE SPIN)

fig = plt.figure(figsize=(10, 7), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
# Set the background color of the 3D plot to white
# Set 3D pane background to white
ax.set_facecolor('white')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.plot(L_vec[:,0], L_vec[:,1], L_vec[:,2], label=r'Trajectory of $\vec{\mathcal{L}}$', color='blue')
ax.plot(S1_vec[:,0], S1_vec[:,1], S1_vec[:,2], label=r'Trajectory of $\vec{\mathcal{S}}_1$', color='green')
ax.plot(S2_vec[:,0], S2_vec[:,1], S2_vec[:,2], label=r'Trajectory of $\vec{\mathcal{S}}_2$', color='orange')
#ax.plot(Xivec_sol[:,0], Xivec_sol[:,1], Xivec_sol[:,2], label=r'Trajectory of $\vec{\chi}_{\mathrm{eff}}$', color='brown')

# Choose an index along the trajectory for the arrow (e.g., halfway)
arrow_scale = 0.05
i_arrow = 0  #len(L_vectors) // 2

# L arrow
L_base = L_vec[i_arrow]
L_dir = L_vec[i_arrow+1] - L_vec[i_arrow]
L_dir = L_dir / np.linalg.norm(L_dir) * arrow_scale
ax.quiver(L_base[0], L_base[1], L_base[2],
          L_dir[0], L_dir[1], L_dir[2],
          color='navy', linewidth=2.5, arrow_length_ratio=0.85, label=r'$\vec{\mathcal{L}}_0$')

# S1 arrow
S1_base = S1_vec[i_arrow]
S1_dir = S1_vec[i_arrow+1] - S1_vec[i_arrow]
S1_dir = S1_dir / np.linalg.norm(S1_dir) * arrow_scale
ax.quiver(S1_base[0], S1_base[1], S1_base[2],
          S1_dir[0], S1_dir[1], S1_dir[2],
          color='purple', linewidth=2.5, arrow_length_ratio=0.85, label=r'$\vec{\mathcal{S}}_{1_0}$')

# S2 arrow
S2_base = S2_vec[i_arrow]
S2_dir = S2_vec[i_arrow+1] - S2_vec[i_arrow]
S2_dir = S2_dir / np.linalg.norm(S2_dir) * arrow_scale
ax.quiver(S2_base[0], S2_base[1], S2_base[2],
          S2_dir[0], S2_dir[1], S2_dir[2],
          color='red', linewidth=2.5, arrow_length_ratio=0.85, label=r'$\vec{\mathcal{S}}_{2_0}$')

Xivec_base = Xivec_sol[i_arrow]
Xivec_dir = Xivec_sol[i_arrow+1] - Xivec_sol[i_arrow]
Xivec_dir = Xivec_dir / np.linalg.norm(Xivec_dir) * arrow_scale
#ax.quiver(Xivec_base[0], Xivec_base[1], Xivec_base[2],
          #Xivec_dir[0], Xivec_dir[1], Xivec_dir[2],
          #color='black', linewidth=2.5, arrow_length_ratio=0.85, label=r'$\vec{\chi}_{\mathrm{eff}_0}$')

ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$y$', fontsize=18)
ax.set_zlabel(r'$z$', fontsize=18)
plt.tick_params(axis='both', labelsize=15)
plt.legend(fontsize=18, loc='upper right')   #, bbox_to_anchor=(0.95, 1.0)
plt.tight_layout()
plt.show()
