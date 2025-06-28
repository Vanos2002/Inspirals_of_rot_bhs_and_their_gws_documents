# SIMULATION OF INSPIRALING COMPACT BINARIES up to 3.5PN order - via Cliﬀord M. Will: Post-Newtonian gravitational radiation and equations of motion via direct integration of the relaxed Einstein equations.
# THIS CODES IMPLEMENTS THE SPIN SUPPLEMENTARY CONDITION (SSC) AS K_{SSC}=1/2
# III. Radiation reaction for binary systems with spinning bodies
# Import of the necessary libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from numba import njit


# Constants (defined for a geometrized unit system)
G = 1                                   # Gravitational constant
c = 1                                   # Speed of light
# Constants in standard SI units
M_sun = 1.98847e30                      # 1 Solar mass in kg
G_SI = 6.6743e-11                       # Gravitational constant in m^3/kg/s^2
c_SI = 299792458                        # Speed of light in m/s


# Set masses in Solar masses
m1_solar = 1                            # Mass of black hole 1 in Solar masses
m2_solar = 1                            # Mass of black hole 2 in Solar masses

M_total_solar = m1_solar + m2_solar     # Total Solar mass of the system


# Black hole masses for the code - (geometrized units)
m1 = m1_solar/M_total_solar             # Mass of black hole 1
m2 = m2_solar/M_total_solar             # Mass of black hole 2

# Convert Solar masses to kg for the code
M_kg = M_total_solar * M_sun                # Total mass in kg
time_unit_seconds = G_SI * M_kg / c_SI**3   # 1 code unit in seconds
sep_unit_meters = G_SI * M_kg / c_SI**2     # 1 code unit in meters

# Derived quantities
M = m1 + m2                                 # Total mass of the system, defined as M=1
mu = m1 * m2 / M                            # Reduced mass
nu = mu / M                                 # Symmetric mass ratio
Mc = (m1 * m2)**(3/5) * (m1 + m2)**(-1/5)   # Chirp mass

rsc = 30 * M  # Initial separation scaling parameter (30 times total sytem mass = 30)


# Initial conditions (ensure non-planar motion)
x1_0 = (m2 / M) * rsc               # Initial x-coordinate of black hole 1
y1_0 = 0                            # Initial y-coordinate of black hole 1
z1_0 = (m2 / M) * rsc / 7           # Initial z-coordinate of black hole 1
x2_0 = -(m1 / M) * rsc              # Initial x-coordinate of black hole 2
y2_0 = 0                            # Initial y-coordinate of black hole 2
z2_0 = -(m1 / M) * rsc / 7          # Initial z-coordinate of black hole 2

# Relative separation of the two black holes
r_sep = np.sqrt((x1_0 - x2_0)**2 + (y1_0 - y2_0)**2 + (z1_0 - z2_0)**2)
v0 = np.sqrt(M / r_sep)             # Newtonian circular velocity (for scaling)
vx1_0 = 0                           # Initial x-velocity of black hole 1
vy1_0 = (m2 / M) * v0               # Initial y-velocity of black hole 1
vz1_0 = 0                           # Initial z-velocity of black hole 1
vx2_0 = 0                           # Initial x-velocity of black hole 2
vy2_0 = -(m1 / M) * v0              # Initial y-velocity of black hole 2
vz2_0 = 0                           # Initial z-velocity of black hole 2

# Spin parameters of individual black holes
chi1 = 1.0
chi2 = 1.0
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


# Time parameters
t_start = 0.0  # Start time
t_end = 500000  # End time
t_steps = 500000  # Number of time steps
t = np.linspace(t_start, t_end, t_steps)  # Time array
# Initial conditions vector
y0 = np.array([x1_0, y1_0, z1_0, vx1_0, vy1_0, vz1_0,
                x2_0, y2_0, z2_0, vx2_0, vy2_0, vz2_0,
                Sp1_0[0], Sp1_0[1], Sp1_0[2], Sp2_0[0], Sp2_0[1], Sp2_0[2]])  # Initial conditions vector
# Solve the ODE
derivs = equations_of_motion()
solution = odeint(derivs, y0, t, atol=1e-13, rtol=1e-13)

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


# Parameters for the animation
num_frames = 10000  # Number of frames
fps = 50           # Frames per second
delta_n = 1690        # Step size

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial plot elements
traj1, = ax.plot([], [], [], lw=1, label="Trajectory of BH1")                 # Trajectory for point 1
traj2, = ax.plot([], [], [], lw=1, label="Trajectory of BH2")                 # Trajectory for point 2
bh1, = ax.plot([], [], [], 'ro', markersize=8 * m1, label="Point 1")          # Point 1
bh2, = ax.plot([], [], [], 'bo', markersize=8 * m2, label="Point 2")          # Point 2
spin1 = None
spin2 = None
total_spin = None
plotTitle = ax.set_title("Inspirals of a binary black hole system up to 3.5PN order")

# Set up 3D plot limits and labels
ax.set_xlim([-30, 30])
ax.set_ylim([-30, 30])
ax.set_zlim([-10, 10])
ax.set_xlabel('X (M)')
ax.set_ylabel('Y (M)')
ax.set_zlabel('Z (M)')

# Initialize the spin vectors
S1v_sol, S2v_sol = [], []
for i in range(len(solution)):
    Sp1_sol = np.array([Sp1x_sol[i], Sp1y_sol[i], Sp1z_sol[i]])
    Sp2_sol = np.array([Sp2x_sol[i], Sp2y_sol[i], Sp2z_sol[i]])
    
    S1v_sol.append(Sp1_sol)
    S2v_sol.append(Sp2_sol)

# Converting lists into numpy arrays
S1v_sol = np.array(S1v_sol)
S2v_sol = np.array(S2v_sol)


# Initialize global variables
spin1, spin2, total_spin = None, None, None
merger_occurred = False
i_final = None

def drawframe(s):
    global spin1, spin2, total_spin
    global merger_occurred, i_final

    i = delta_n * s + 1

    # Check if merger has occurred
    if not merger_occurred:
        dist = np.sqrt((x1_sol[i - 1] - x2_sol[i - 1])**2 + 
                       (y1_sol[i - 1] - y2_sol[i - 1])**2 + 
                       (z1_sol[i - 1] - z2_sol[i - 1])**2)
        
        rvec = np.array([x1_sol[i] - x2_sol[i], y1_sol[i] - y2_sol[i], z1_sol[i] - z2_sol[i]])
        vvec = np.array([vx1_sol[i] - vx2_sol[i], vy1_sol[i] - vy2_sol[i], vz1_sol[i] - vz2_sol[i]])
        L = reduced_angular_momentum(rvec, vvec)
        Sp1_sol = np.array([Sp1x_sol[i], Sp1y_sol[i], Sp1z_sol[i]])  # Proper spin of black hole 1
        Sp2_sol = np.array([Sp2x_sol[i], Sp2y_sol[i], Sp2z_sol[i]])  # Proper spin of black hole 2
        Sp_sol = Sp1_sol + Sp2_sol                                   # Total proper spin vector

        # Compute effective spin aligned with L
        L_hat = L / (np.linalg.norm(L) + 1e-16)
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

        if dist < 2 * r_ISCO:  # Threshold distance for merger (double ISCO)
            merger_occurred = True
            i_final = i  # Store final frame index
            print(f"Merger detected at t = {t[i_final]:.4f}, r_ISCO (M) = {dist / 2:.3f}")
            print(f"Merger detected (real time): {t[i_final] * time_unit_seconds:.4f} s")
            print(f"KERR ISCO radius (meters): r_ISCO ={dist * sep_unit_meters / 2:.3f}")
            return traj1, traj2, bh1, bh2, spin1, spin2, total_spin

    if merger_occurred:
        i = i_final  # Freeze black hole positions at merger frame

    # Update trajectories (stops updating after merger)
    traj1.set_data(x1_sol[:i], y1_sol[:i])
    traj1.set_3d_properties(z1_sol[:i])
    bh1.set_data([x1_sol[i - 1]], [y1_sol[i - 1]])
    bh1.set_3d_properties([z1_sol[i - 1]])

    traj2.set_data(x2_sol[:i], y2_sol[:i])
    traj2.set_3d_properties(z2_sol[:i])
    bh2.set_data([x2_sol[i - 1]], [y2_sol[i - 1]])
    bh2.set_3d_properties([z2_sol[i - 1]])

    # Remove old spin vectors
    if spin1:
        spin1.remove()
        spin1 = None
    if spin2:
        spin2.remove()
        spin2 = None
    if total_spin:
        total_spin.remove()
        total_spin = None

    if merger_occurred:
        # Determine the remaining black hole
        if m1 < m2:
            bh1.set_visible(False)
            x_remain, y_remain, z_remain = x2_sol[i_final - 1], y2_sol[i_final - 1], z2_sol[i_final - 1]
        else:
            bh2.set_visible(False)
            x_remain, y_remain, z_remain = x1_sol[i_final - 1], y1_sol[i_final - 1], z1_sol[i_final - 1]

        # Compute total spin
        S_total = S1v_sol[i_final - 1] + S2v_sol[i_final - 1]
        spin_length = np.linalg.norm(S_total)  # Adjust vector length dynamically

        # Plot total spin vector if it is not zero
        if spin_length > 0:
            total_spin = ax.quiver(
                x_remain, y_remain, z_remain,
                S_total[0], S_total[1], S_total[2],
                color='purple', length=spin_length, normalize=True
            )
    else:
        # Before merger, show individual spin vectors
        spin1 = ax.quiver(x1_sol[i - 1], y1_sol[i - 1], z1_sol[i - 1], 
                          S1v_sol[i - 1, 0], S1v_sol[i - 1, 1], S1v_sol[i - 1, 2], 
                          color='r', length=5, normalize=True)
        spin2 = ax.quiver(x2_sol[i - 1], y2_sol[i - 1], z2_sol[i - 1], 
                          S2v_sol[i - 1, 0], S2v_sol[i - 1, 1], S2v_sol[i - 1, 2], 
                          color='b', length=5, normalize=True)

    return traj1, traj2, bh1, bh2, spin1, spin2, total_spin

# Create the animation
anim = FuncAnimation(fig, drawframe, frames=num_frames, interval=1000 // fps)

# Display the animation
plt.show()
