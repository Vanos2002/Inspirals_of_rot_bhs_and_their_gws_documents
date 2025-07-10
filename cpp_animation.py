import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Loads the data from the CSV file
data = np.loadtxt("orbit.csv", delimiter=",")
x1, y1, z1 = data[:,0], data[:,1], data[:,2]
x2, y2, z2 = data[:,3], data[:,4], data[:,5]
vx1, vy1, vz1 = data[:,6], data[:,7], data[:,8]
vx2, vy2, vz2 = data[:,9], data[:,10], data[:,11]
Sp1x, Sp1y, Sp1z = data[:,12], data[:,13], data[:,14]
Sp2x, Sp2y, Sp2z = data[:,15], data[:,16], data[:,17]

m1 = data[0, -2]  # Mass of BH1 (we assume constant throughout the simulation)
m2 = data[0, -1]  # Mass of BH2 (we assume constant throughout the simulation)

M = 1             # Total mass of the system (normalized to 1 for simplicity)
# Animation parameters
num_frames = 1000       # Generally 500–2000 for smoothness
delta_n = max(1, len(x1) // num_frames)
fps = 60                # Frames per second for the animation

margin = 2              # Add a little margin for aesthetics

x_min = min(np.min(x1), np.min(x2)) - margin
x_max = max(np.max(x1), np.max(x2)) + margin
y_min = min(np.min(y1), np.min(y2)) - margin
y_max = max(np.max(y1), np.max(y2)) + margin
z_min = min(np.min(z1), np.min(z2)) - margin
z_max = max(np.max(z1), np.max(z2)) + margin

# Kerr ISCO function
def KerrISCO(M, chi_eff_kerr, Z1, Z2, sign=1):
    return M * (3 + Z2 + sign * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)))

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
traj1, = ax.plot([], [], [], lw=1, label="Trajectory of BH1")
traj2, = ax.plot([], [], [], lw=1, label="Trajectory of BH2")
bh1, = ax.plot([], [], [], 'ro', markersize=16 * m1, label="Point 1")
bh2, = ax.plot([], [], [], 'bo', markersize=16 * m2 , label="Point 2")
spin1 = None
spin2 = None
total_spin = None
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_zlim([z_min, z_max])
ax.set_xlabel('X (M)')
ax.set_ylabel('Y (M)')
ax.set_zlabel('Z (M)')
ax.set_title("Inspirals of a binary black hole system up to 3.5PN order")

# Prepare spin arrays
S1v_sol = np.column_stack((Sp1x, Sp1y, Sp1z))
S2v_sol = np.column_stack((Sp2x, Sp2y, Sp2z))

# Animation state
merger_occurred = False
i_final = None

def reduced_angular_momentum(rvec, vvec):
    return np.cross(rvec, vvec)

def drawframe(s):
    global spin1, spin2, total_spin, merger_occurred, i_final

    i = s * delta_n
    if i >= len(x1):
        i = len(x1) - 1

    # Check if merger has occurred (essential for drawing the spin vectors and inspirals)
    if not merger_occurred:
        dist = np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2 + (z1[i] - z2[i])**2)
        rvec = np.array([x1[i] - x2[i], y1[i] - y2[i], z1[i] - z2[i]])
        vvec = np.array([vx1[i] - vx2[i], vy1[i] - vy2[i], vz1[i] - vz2[i]])
        L = reduced_angular_momentum(rvec, vvec)
        Sp1_sol = S1v_sol[i]
        Sp2_sol = S2v_sol[i]
        Sp_sol = Sp1_sol + Sp2_sol

        # Compute effective spin aligned with L
        L_hat = L / (np.linalg.norm(L) + 1e-16)
        S_tot = Sp_sol
        chi_eff_kerr = np.dot(S_tot, L_hat) / M**2
        chi_eff_kerr = np.clip(chi_eff_kerr, -1.0, 1.0)

        # Determine ISCO direction: prograde (sign=-1), retrograde (sign=+1)
        LS_dot = np.dot(L, S_tot)
        sign = -1 if LS_dot > 0 else +1

        # Kerr ISCO auxiliary quantities
        Z1 = 1 + (1 - chi_eff_kerr**2)**(1/3) * (
                (1 + chi_eff_kerr)**(1/3) + (1 - chi_eff_kerr)**(1/3))
        Z2 = np.sqrt(3 * chi_eff_kerr**2 + Z1**2)

        # ISCO radius
        r_ISCO = KerrISCO(M, chi_eff_kerr, Z1, Z2, sign=sign)

        if dist < 2 * r_ISCO:
            merger_occurred = True
            i_final = i

    if merger_occurred:
        i = i_final

    # Update trajectories
    traj1.set_data(x1[:i], y1[:i])
    traj1.set_3d_properties(z1[:i])
    bh1.set_data([x1[i]], [y1[i]])
    bh1.set_3d_properties([z1[i]])
    traj2.set_data(x2[:i], y2[:i])
    traj2.set_3d_properties(z2[:i])
    bh2.set_data([x2[i]], [y2[i]])
    bh2.set_3d_properties([z2[i]])

    # Remove old spin vectors
    global spin1, spin2, total_spin
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
        # Show total spin at merger
        S_total = S1v_sol[i] + S2v_sol[i]
        spin_length = np.linalg.norm(S_total)
        if spin_length > 0:
            total_spin = ax.quiver(
                x1[i], y1[i], z1[i],
                S_total[0], S_total[1], S_total[2],
                color='purple', length=spin_length, normalize=True
            )
    else:
        # Show individual spins
        spin1 = ax.quiver(x1[i], y1[i], z1[i],
                          S1v_sol[i, 0], S1v_sol[i, 1], S1v_sol[i, 2],
                          color='r', length=5, normalize=True)
        spin2 = ax.quiver(x2[i], y2[i], z2[i],
                          S2v_sol[i, 0], S2v_sol[i, 1], S2v_sol[i, 2],
                          color='b', length=5, normalize=True)

    return traj1, traj2, bh1, bh2, spin1, spin2, total_spin

anim = FuncAnimation(fig, drawframe, frames=num_frames, interval=1000 // fps)
plt.legend()
plt.show()
