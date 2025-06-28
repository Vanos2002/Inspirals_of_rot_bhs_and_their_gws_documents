# SIMULATION OF A BINARY BLACK HOLE SYSTEM UP TO THE SECOND POST-NEWTONIAN ORDER
# Import of the necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator, FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from joblib import Parallel, delayed


mpl.rcParams['agg.path.chunksize'] = 100000  # We employ for high resolution matplotlib plotting
np.set_printoptions(precision=15)   # Printing values will be evaluated to 15 digits

# Constants (changed for a geometrized unit system)
G = 1.0         # Gravitational constant
c = 1.0       # Speed of light

# Black hole masses (geometrized units)
m1 = 0.5   # Mass of black hole 1
m2 = 0.5   # Mass of black hole 2

q = m1 / m2  # Mass ratio of the two black holes

# Derived quantities
M = m1 + m2  # Total mass of the system
mu = m1 * m2 / M  # Reduced mass
nu = mu / M   # Symmetric mass ratio
GM = G * M   # Abbreviation for a frequent product term

# Spin parameters
chi1 = -0.6  # Dimensionless spin parameter for the first black hole
chi2 = 1.0   # Dimensionless spin parameter for the second black hole

# Effective spin constants (employed for effective spin definition)
sigma1 = 1.0 + 3 * m2 / (4 * m1)
sigma2 = 1.0 + 3 * m1 / (4 * m2)

# Total spin magnitude
S1m = chi1 * G * m1**2 / c
S2m = chi2 * G * m2**2 / c



# INITIAL CONDITIONS IN THE COM FRAME
r_0 = 30   # Relative separation (in M)
pr_0 = 2e-16    # Radial momentum (conjugate to r) - (arbitrary units)

# Initial spin vector components (ensure its normalization equals S1m, respect. S2m)
S1x0 = S1m * 2/3 # Initial x-component of spin vector for black hole 1
S1y0 = -S1m * 1/3  # Initial y-component of spin vector for black hole 1
S1z0 = -S1m  * 2/3  # Initial z-component of spin vector for black hole 1

S2x0 = -S2m * 2/7   # Initial x-component of spin vector for black hole 2
S2y0 = -S2m * 3/7 # Initial y-component of spin vector for black hole 2
S2z0 = S2m * 6/7  # Initial z-component of spin vector for black hole 2

# Initial orbital angular momentum vector components
Lx0 = 0.0  # Initial x-component of angular momentum
Ly0 = 0.0    # Initial y-component of angular momentum
Lz0 = mu * np.sqrt(r_0)         # Initial y-component of angular momentum

# Constants in standardized units
G_r = 6.67430e-11        # m^3 kg^-1 s^-2 - Gravitational constant
c_r = 299792458          # m/s - Speed of light
M_solar = 1.98847e30     # kg - Solar mass

# Total system mass (Real physical mass)
M_r = 60 * M_solar       # kg

# Relative separation calculation to SI units
r_sep_phys = r_0 * G_r * M_r / c_r**2
print("Initial separation in physical units (m):", r_sep_phys)


# FUNCTION DEFINITIONS
def spin_vectors(S1x, S1y, S1z, S2x, S2y, S2z):

    S1 = np.array([S1x, S1y, S1z])
    S2 = np.array([S2x, S2y, S2z])

    return S1, S2


def angular_momentum(Lx, Ly, Lz):
    
    L = np.array([Lx, Ly, Lz])
    
    return L

def angular_momentum_magnitude(L):
    return np.linalg.norm(L)

def total_angular_momentum(Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z):
    return np.array([Lx, Ly, Lz]) + np.array([S1x, S1y, S1z]) + np.array([S2x, S2y, S2z])


# Defining the Hamiltonian dynamics up to the second post-Newtonian order - we define each term individually for the possibility of examining up to a certain level
def HN(r, pr, Lx, Ly, Lz):
    
    return (1/2)*(pr**2 + (-2*r + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*mu**2))/r**2)*mu
    
def H1PN(r, pr, Lx, Ly, Lz):
    return (mu*(4/r**2 + (pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))**2*(-1 + 3*nu) - 
    (4*(((Lx**2 + Ly**2 + Lz**2)*(3 + nu))/(G**2*M**2*r**2*mu**2) + pr**2*(3 + 2*nu)))/r))/(8*c**2)

def H1_5PN(r, Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z):
    return (2*(Lx*(S1x + (3*m2*S1x)/(4*m1) + S2x + (3*m1*S2x)/(4*m2)) + 
    Ly*(S1y + (3*m2*S1y)/(4*m1) + S2y + (3*m1*S2y)/(4*m2)) + 
    Lz*(S1z + (3*m2*S1z)/(4*m1) + S2z + (3*m1*S2z)/(4*m2))))/(c**2*G**2*M**3*r**3)
  
def H2PN(r, pr, Lx, Ly, Lz):
    
    return (1/(16*c**4))*(mu*(-((4*(1 + 3*nu))/r**3) + 
    (pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))**3*(1 + 5*(-1 + nu)*nu) + 
    (8*(((Lx**2 + Ly**2 + Lz**2)*(5 + 8*nu))/(G**2*M**2*r**2*mu**2) + pr**2*(5 + 11*nu)))/r**2 + 
    (2*(-5*pr**4*nu**2 - (2*(Lx**2 + Ly**2 + Lz**2)*pr**2*nu**2)/(G**2*M**2*r**2*mu**2) + 
       (pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))**2*(5 - nu*(20 + 3*nu))))/r))

def HS1S1(r, Lx, Ly, Lz, S1x, S1y, S1z):

    return -((m2*(-6*Ly*Lz*S1y*S1z - 6*Lx*S1x*(Ly*S1y + Lz*S1z) + Lz**2*(S1x**2 + S1y**2 - 2*S1z**2) + 
     Ly**2*(S1x**2 - 2*S1y**2 + S1z**2) + Lx**2*(-2*S1x**2 + S1y**2 + S1z**2)))/
    (2*c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m1*r**3))


def HS2S2(r, Lx, Ly, Lz, S2x, S2y, S2z):

    return -((m1*(-6*Ly*Lz*S2y*S2z - 6*Lx*S2x*(Ly*S2y + Lz*S2z) + Lz**2*(S2x**2 + S2y**2 - 2*S2z**2) + 
     Ly**2*(S2x**2 - 2*S2y**2 + S2z**2) + Lx**2*(-2*S2x**2 + S2y**2 + S2z**2)))/
    (2*c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m2*r**3))
    
    
def HS1S2(r, Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z):

    return (3*Lx*(Ly*S1y*S2x + Lz*S1z*S2x + Ly*S1x*S2y + Lz*S1x*S2z) + 
    3*Ly*Lz*(S1z*S2y + S1y*S2z) - Lz**2*(S1x*S2x + S1y*S2y - 2*S1z*S2z) + 
    Lx**2*(2*S1x*S2x - S1y*S2y - S1z*S2z) - Ly**2*(S1x*S2x - 2*S1y*S2y + S1z*S2z))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**3)


def Hamiltonian(r, pr, Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z):
    return (HN(r, pr, Lx, Ly, Lz)
        + H1PN(r, pr, Lx, Ly, Lz)
        + H1_5PN(r, Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z)
        + H2PN(r, pr, Lx, Ly, Lz)
        + HS1S1(r, Lx, Ly, Lz, S1x, S1y, S1z)
        + HS2S2(r, Lx, Ly, Lz, S2x, S2y, S2z)
        + HS1S2(r, Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z))


# Equations (EoM) that govern the motion of the binary black hole system's center of mass (derived in Mathematica - via 
def equations_of_motion(tau, state):
    r, pr, Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z = state
    
    #DPRDTAU TERMS

    dprdtau_newtonian = (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**3*mu) - mu/r**2

    dprdtau_1PN = -((mu*(-2*r + (3*(Lx**2 + Ly**2 + Lz**2)*(3 + nu))/(G**2*M**2*mu**2) + pr**2*r**2*(3 + 2*nu) - 
     ((Lx**2 + Ly**2 + Lz**2)*(Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)*(-1 + 3*nu))/
      (G**4*M**4*r*mu**4)))/(2*c**2*r**4))
    
    dprdtau_15PN = (6*(Lx*(S1x + (3*m2*S1x)/(4*m1) + S2x + (3*m1*S2x)/(4*m2)) + 
    Ly*(S1y + (3*m2*S1y)/(4*m1) + S2y + (3*m1*S2y)/(4*m2)) + 
    Lz*(S1z + (3*m2*S1z)/(4*m1) + S2z + (3*m1*S2z)/(4*m2))))/(c**2*G**2*M**3*r**4)
    
    dprdtau_2PN = -((1/(8*c**4*r**5))*(mu*(5*pr**4*r**3*nu**2 + (2*(Lx**2 + Ly**2 + Lz**2)*pr**2*r*nu**2)/
      (G**2*M**2*mu**2) + 6*r*(1 + 3*nu) - (16*(Lx**2 + Ly**2 + Lz**2)*(5 + 8*nu))/
      (G**2*M**2*mu**2) - 8*pr**2*r**2*(5 + 11*nu) - 
     (3*(Lx**2 + Ly**2 + Lz**2)*(Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)**2*
       (1 + 5*(-1 + nu)*nu))/(G**6*M**6*r**2*mu**6) + 
     ((Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)**2*(-5 + nu*(20 + 3*nu)))/
      (G**4*M**4*r*mu**4) + (4*(Lx**2 + Ly**2 + Lz**2)*(G**2*M**2*pr**2*r**2*mu**2*nu**2 + 
        (Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)*(-5 + nu*(20 + 3*nu))))/
      (G**4*M**4*r*mu**4))))

    dprdtau_S1S1 = -((3*m2*(-6*Ly*Lz*S1y*S1z - 6*Lx*S1x*(Ly*S1y + Lz*S1z) + 
     Lz**2*(S1x**2 + S1y**2 - 2*S1z**2) + Ly**2*(S1x**2 - 2*S1y**2 + S1z**2) + 
     Lx**2*(-2*S1x**2 + S1y**2 + S1z**2)))/(2*c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m1*r**4))
    
    dprdtau_S2S2 = -((3*m1*(-6*Ly*Lz*S2y*S2z - 6*Lx*S2x*(Ly*S2y + Lz*S2z) + 
     Lz**2*(S2x**2 + S2y**2 - 2*S2z**2) + Ly**2*(S2x**2 - 2*S2y**2 + S2z**2) + 
     Lx**2*(-2*S2x**2 + S2y**2 + S2z**2)))/(2*c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m2*r**4))
    
    dprdtau_S1S2 = -((3*(-3*Lx*(Ly*S1y*S2x + Lz*S1z*S2x + Ly*S1x*S2y + Lz*S1x*S2z) - 
     3*Ly*Lz*(S1z*S2y + S1y*S2z) + Lz**2*(S1x*S2x + S1y*S2y - 2*S1z*S2z) + 
     Ly**2*(S1x*S2x - 2*S1y*S2y + S1z*S2z) + Lx**2*(-2*S1x*S2x + S1y*S2y + S1z*S2z)))/
    (c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**4))


    #DRDTAU TERMS
    
    drdtau_newtonian = mu * pr

    drdtau_1PN = (mu*(-((8*pr*(3 + 2*nu))/r) + 4*pr*(pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))*
     (-1 + 3*nu)))/(8*c**2)

    drdtau_15PN = 0

    drdtau_2PN = (pr*mu*((40 + 88*nu)/r**2 + 3*(pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))**2*
     (1 + 5*(-1 + nu)*nu) - (4*(Lx**2 + Ly**2 + Lz**2)*(-5 + 4*nu*(5 + nu)))/
     (G**2*M**2*r**3*mu**2) + (4*pr**2*(5 - 4*nu*(5 + 2*nu)))/r))/(8*c**4)
   
    drdtau_S1S1 = 0

    drdtau_S2S2 = 0

    drdtau_S1S2 = 0


    # ANGULAR MOMENTUM TERMS

    OmegaL_newtonian = np.array([Lx/(G**2*M**2*r**2*mu), Ly/(G**2*M**2*r**2*mu), Lz/(G**2*M**2*r**2*mu)])
    
    OmegaL_1PN = np.array([(Lx*(-2*(3 + nu) + r*(pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))*(-1 + 3*nu)))/(2*c**2*G**2*M**2*r**3*mu), 
    (Ly*(-2*(3 + nu) + r*(pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))*(-1 + 3*nu)))/(2*c**2*G**2*M**2*r**3*mu), 
    (Lz*(-2*(3 + nu) + r*(pr**2 + (Lx**2 + Ly**2 + Lz**2)/(G**2*M**2*r**2*mu**2))*(-1 + 3*nu)))/(2*c**2*G**2*M**2*r**3*mu)])
    
    OmegaL_15PN = np.array([(2*(S1x + (3*m2*S1x)/(4*m1) + S2x + (3*m1*S2x)/(4*m2)))/(c**2*G**2*M**3*r**3), 
    (2*(S1y + (3*m2*S1y)/(4*m1) + S2y + (3*m1*S2y)/(4*m2)))/(c**2*G**2*M**3*r**3), 
    (2*(S1z + (3*m2*S1z)/(4*m1) + S2z + (3*m1*S2z)/(4*m2)))/(c**2*G**2*M**3*r**3)])

    OmegaL_2PN = np.array([(1/(8*c**4*G**6*M**6*r**6*mu**5))*(Lx*(-4*G**4*M**4*pr**2*r**3*mu**4*nu**2 + 
     8*G**4*M**4*r**2*mu**4*(5 + 8*nu) + 3*(Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)**2*
      (1 + 5*(-1 + nu)*nu) + 4*G**2*M**2*r*mu**2*(Lx**2 + Ly**2 + Lz**2 + 
       G**2*M**2*pr**2*r**2*mu**2)*(5 - nu*(20 + 3*nu)))), 
    (1/(8*c**4*G**6*M**6*r**6*mu**5))*(Ly*(-4*G**4*M**4*pr**2*r**3*mu**4*nu**2 + 
     8*G**4*M**4*r**2*mu**4*(5 + 8*nu) + 3*(Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)**2*(1 + 5*(-1 + nu)*nu) 
     + 4*G**2*M**2*r*mu**2*(Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)*(5 - nu*(20 + 3*nu)))), 
    (1/(8*c**4*G**6*M**6*r**6*mu**5))*(Lz*(-4*G**4*M**4*pr**2*r**3*mu**4*nu**2 + 
     8*G**4*M**4*r**2*mu**4*(5 + 8*nu) + 3*(Lx**2 + Ly**2 + Lz**2 + G**2*M**2*pr**2*r**2*mu**2)**2*(1 + 5*(-1 + nu)*nu) + 4*G**2*M**2*r*mu**2*(Lx**2 + Ly**2 + Lz**2 + 
       G**2*M**2*pr**2*r**2*mu**2)*(5 - nu*(20 + 3*nu))))])
    
    OmegaL_S1S1 = np.array([-((3*m2*(Lx*S1x + Ly*S1y + Lz*S1z)*((-Ly**2)*S1x + Lx*Ly*S1y + Lz*((-Lz)*S1x + Lx*S1z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*m1*r**3)), 
    (3*m2*(Lx*S1x + Ly*S1y + Lz*S1z)*((-Lx)*Ly*S1x + Lx**2*S1y + Lz*(Lz*S1y - Ly*S1z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*m1*r**3), 
    (3*m2*((-Lz)*(Lx*S1x + Ly*S1y) + (Lx**2 + Ly**2)*S1z)*(Lx*S1x + Ly*S1y + Lz*S1z))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*m1*r**3)])
    
    OmegaL_S2S2 = np.array([-((3*m1*(Lx*S2x + Ly*S2y + Lz*S2z)*((-Ly**2)*S2x + Lx*Ly*S2y + Lz*((-Lz)*S2x + Lx*S2z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*m2*r**3)), 
    (3*m1*(Lx*S2x + Ly*S2y + Lz*S2z)*((-Lx)*Ly*S2x + Lx**2*S2y + Lz*(Lz*S2y - Ly*S2z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*m2*r**3), 
    (3*m1*((-Lz)*(Lx*S2x + Ly*S2y) + (Lx**2 + Ly**2)*S2z)*(Lx*S2x + Ly*S2y + Lz*S2z))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*m2*r**3)])
    
    OmegaL_S1S2 = np.array([(-3*Lx**2*(Ly*S1y*S2x + Lz*S1z*S2x + Ly*S1x*S2y + Lz*S1x*S2z) + 
    3*(Ly**2 + Lz**2)*(Ly*S1y*S2x + Lz*S1z*S2x + Ly*S1x*S2y + Lz*S1x*S2z) + 
    6*Lx*(Ly**2*(S1x*S2x - S1y*S2y) - Ly*Lz*(S1z*S2y + S1y*S2z) + 
      Lz**2*(S1x*S2x - S1z*S2z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*r**3), 
    (3*(Lx**3*(S1y*S2x + S1x*S2y) - Lx*Ly**2*(S1y*S2x + S1x*S2y) + 
     Lx*Lz**2*(S1y*S2x + S1x*S2y) - 2*Lx*Ly*Lz*(S1z*S2x + S1x*S2z) - 
     Ly**2*Lz*(S1z*S2y + S1y*S2z) + Lz**3*(S1z*S2y + S1y*S2z) + 
     Lx**2*(-2*Ly*S1x*S2x + 2*Ly*S1y*S2y + Lz*S1z*S2y + Lz*S1y*S2z) + 
     2*Ly*Lz**2*(S1y*S2y - S1z*S2z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*r**3), 
    (3*(-2*Lz*(Lx*S1x + Ly*S1y) + (Lx**2 + Ly**2 - Lz**2)*S1z)*(Lx*S2x + Ly*S2y) + 
    3*((Lx**2 + Ly**2 - Lz**2)*(Lx*S1x + Ly*S1y) + 2*(Lx**2 + Ly**2)*Lz*S1z)*S2z)/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)**2*M**3*r**3)])
    
    dLdtau_newtonian = np.cross(OmegaL_newtonian, np.array([Lx, Ly, Lz]))
    
    dLdtau_1PN = np.cross(OmegaL_1PN, np.array([Lx, Ly, Lz]))
    
    dLdtau_15PN = np.cross(OmegaL_15PN, np.array([Lx, Ly, Lz]))
    
    dLdtau_2PN = np.cross(OmegaL_2PN, np.array([Lx, Ly, Lz]))
    
    dLdtau_S1S1 = np.cross(OmegaL_S1S1, np.array([Lx, Ly, Lz]))
    
    dLdtau_S2S2 = np.cross(OmegaL_S2S2, np.array([Lx, Ly, Lz]))
    
    dLdtau_S1S2 = np.cross(OmegaL_S1S2, np.array([Lx, Ly, Lz]))
    
    
    #SPIN TERMS
    
    dS1dtau_newtonian = np.array([0, 0, 0])
    dS2dtau_newtonian = np.array([0, 0, 0])
    
    
    dS1dtau_1PN = np.array([0, 0, 0])
    dS2dtau_1PN = np.array([0, 0, 0])
    
    OmegaS1_15PN = np.array([(2*Lx*(1 + (3*m2)/(4*m1)))/(c**2*G**2*M**3*r**3), (2*Ly*(1 + (3*m2)/(4*m1)))/(c**2*G**2*M**3*r**3), (2*Lz*(1 + (3*m2)/(4*m1)))/(c**2*G**2*M**3*r**3)])
    
    OmegaS2_15PN = np.array([(2*Lx*(1 + (3*m1)/(4*m2)))/(c**2*G**2*M**3*r**3), (2*Ly*(1 + (3*m1)/(4*m2)))/(c**2*G**2*M**3*r**3), (2*Lz*(1 + (3*m1)/(4*m2)))/(c**2*G**2*M**3*r**3)])
    

    dS1dtau_15PN = np.cross(OmegaS1_15PN, np.array([S1x, S1y, S1z]))

    dS2dtau_15PN = np.cross(OmegaS2_15PN, np.array([S2x, S2y, S2z]))

    dS1dtau_2PN = np.array([0, 0, 0])
    
    dS2dtau_2PN = np.array([0, 0, 0])
    
    OmegaS1_S1S1 = np.array([(m2*(2*Lx**2*S1x - (Ly**2 + Lz**2)*S1x + 3*Lx*(Ly*S1y + Lz*S1z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m1*r**3), 
    (m2*(3*Lx*Ly*S1x - Lx**2*S1y + 2*Ly**2*S1y - Lz**2*S1y + 3*Ly*Lz*S1z))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m1*r**3), 
    (3*Lz*m2*(Lx*S1x + Ly*S1y) - (Lx**2 + Ly**2 - 2*Lz**2)*m2*S1z)/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m1*r**3)])


    dS1dtau_S1S1 = np.cross(OmegaS1_S1S1, np.array([S1x, S1y, S1z]))
    
    dS2dtau_S1S1 = np.array([0, 0, 0])
    
    OmegaS2_S2S2 = np.array([(m1*(2*Lx**2*S2x - (Ly**2 + Lz**2)*S2x + 3*Lx*(Ly*S2y + Lz*S2z)))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m2*r**3), 
    (m1*(3*Lx*Ly*S2x - Lx**2*S2y + 2*Ly**2*S2y - Lz**2*S2y + 3*Ly*Lz*S2z))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m2*r**3), 
    (3*Lz*m1*(Lx*S2x + Ly*S2y) - (Lx**2 + Ly**2 - 2*Lz**2)*m1*S2z)/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*m2*r**3)])
    
    dS1dtau_S2S2 = np.array([0, 0, 0])
    
    dS2dtau_S2S2 = np.cross(OmegaS2_S2S2, np.array([S2x, S2y, S2z]))
    
    OmegaS1_S1S2 = np.array([(2*Lx**2*S2x - (Ly**2 + Lz**2)*S2x + 3*Lx*(Ly*S2y + Lz*S2z))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**3), 
    (3*Lx*Ly*S2x - Lx**2*S2y + 2*Ly**2*S2y - Lz**2*S2y + 3*Ly*Lz*S2z)/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**3), 
    (3*Lz*(Lx*S2x + Ly*S2y) - (Lx**2 + Ly**2 - 2*Lz**2)*S2z)/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**3)])
    
    OmegaS2_S1S2 = np.array([(2*Lx**2*S1x - (Ly**2 + Lz**2)*S1x + 3*Lx*(Ly*S1y + Lz*S1z))/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**3), 
    (3*Lx*Ly*S1x - Lx**2*S1y + 2*Ly**2*S1y - Lz**2*S1y + 3*Ly*Lz*S1z)/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**3), 
    (3*Lz*(Lx*S1x + Ly*S1y) - (Lx**2 + Ly**2 - 2*Lz**2)*S1z)/(c**2*G**2*(Lx**2 + Ly**2 + Lz**2)*M**3*r**3)])
    
    dS1dtau_S1S2 = np.cross(OmegaS1_S1S2, np.array([S1x, S1y, S1z]))

    dS2dtau_S1S2 = np.cross(OmegaS2_S1S2, np.array([S2x, S2y, S2z]))

    
    # Summation of individual terms (remove respective PN terms for different PN examination)
    drdtau = drdtau_newtonian + drdtau_1PN + drdtau_15PN + drdtau_2PN + drdtau_S1S1 + drdtau_S2S2 + drdtau_S1S2

    dprdtau = dprdtau_newtonian + dprdtau_1PN + dprdtau_15PN + dprdtau_2PN + dprdtau_S1S1 + dprdtau_S2S2 + dprdtau_S1S2

    dLdtau = dLdtau_newtonian + dLdtau_1PN + dLdtau_15PN + dLdtau_2PN + dLdtau_S1S1 + dLdtau_S2S2 + dLdtau_S1S2

    dS1dtau = dS1dtau_newtonian + dS1dtau_1PN + dS1dtau_15PN + dS1dtau_2PN + dS1dtau_S1S1 + dS1dtau_S2S2 + dS1dtau_S1S2

    dS2dtau = dS2dtau_newtonian + dS2dtau_1PN + dS2dtau_15PN + dS2dtau_2PN + dS2dtau_S1S1 + dS2dtau_S2S2 + dS2dtau_S1S2

    return [drdtau, dprdtau, dLdtau[0], dLdtau[1], dLdtau[2], dS1dtau[0], dS1dtau[1], dS1dtau[2], dS2dtau[0], dS2dtau[1], dS2dtau[2]]


# Define the initial state and time span
initial_state = [r_0, pr_0, Lx0, Ly0, Lz0, S1x0, S1y0, S1z0, S2x0, S2y0, S2z0]

Tp = 2 * np.pi * np.sqrt(r_0**3 * GM**2)  # Orbital period (one Newtonian)
t_span = (0, 100 * Tp)  # Start and end times
tau = np.linspace(0, 100 * Tp, 1000000)  # Time points for evaluation

# Ensure tau is within t_span
tau = np.clip(tau, t_span[0], t_span[1])

# Solving the equations of motion with an 8th order RK SciPy integrator
solution = solve_ivp(
    equations_of_motion,
    t_span,
    initial_state,
    method='DOP853',
    atol=1e-18,
    rtol=2.23e-14,
    t_eval=tau
)
r_sol, pr_sol, Lx_sol, Ly_sol, Lz_sol, S1x_sol, S1y_sol, S1z_sol, S2x_sol, S2y_sol, S2z_sol = solution.y

# Ensure different quantities have the same length
print("len(tau):", len(tau))
print("len(r_sol):", len(r_sol))
print("len(pr_sol):", len(pr_sol))
print("len(Lx_sol):", len(Lx_sol))
print("len(Ly_sol):", len(Ly_sol))
print("len(Lz_sol):", len(Lz_sol))
print("len(S1x_sol):", len(S1x_sol))
print("len(S2x_sol):", len(S2x_sol))
print("len(S1y_sol):", len(S1y_sol))
print("len(S2y_sol):", len(S2y_sol))
print("len(S1z_sol):", len(S1z_sol))
print("len(S2z_sol):", len(S2z_sol))

# Set a global labelsize for plots
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# Extract the spin vectors
S1_vectors = np.array([
    [S1x_sol[i], S1y_sol[i], S1z_sol[i]]
    for i in range(len(tau))
])

S2_vectors = np.array([
    [S2x_sol[i], S2y_sol[i], S2z_sol[i]]
    for i in range(len(tau))
])
S1_mag = np.linalg.norm(S1_vectors, axis=1)
S2_mag = np.linalg.norm(S2_vectors, axis=1)
S1_vectors = S1_vectors * (S1m / S1_mag[:, np.newaxis])
S2_vectors = S2_vectors * (S2m / S2_mag[:, np.newaxis])
S_mag = np.linalg.norm(S1_vectors + S2_vectors, axis=1)

# Relative error of spins
rel_err_S1 = np.abs(S1_mag - S1m) / np.abs(S1m)
rel_err_S2 = np.abs(S2_mag - S2m) / np.abs(S2m)

# Plot together the individual spin magnitudes and the total spin magnitude
plt.figure(figsize=(10,6))
plt.plot(tau, S1_mag, label=r'$|\mathbf{S}_1|$')
plt.plot(tau, S2_mag, label=r'$|\mathbf{S}_2|$')
plt.plot(tau, S_mag, label=r'$|\mathbf{S}_1+\mathbf{S}_2|$', linestyle='--', color='k')
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel('Total Spin Magnitudes')
plt.title('Total Spin Magnitudes vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Conservation of individual spin magnitudes check
plt.figure(figsize=(10,6))
plt.ylim(1e-16, 1e-13)
plt.xlim(tau[0], tau[-1])
plt.plot(tau, rel_err_S1, label=r'$\delta_{||\vec{\mathcal{S}}_1||}$')
plt.plot(tau, rel_err_S2, label=r'$\delta_{||\vec{\mathcal{S}}_2||}$')
plt.yscale('log')
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel(r'$\delta_{||\vec{\mathcal{S}}_i||}$', fontsize=18)
#plt.title('Relative Error of Spin Magnitudes vs Time', fontsize=18)
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Examination of spin magnitude relative error - Note this quantity is necessarily NOT a conservative quantity, rather a measure of symmetry
S_mag0 = S_mag[0]
rel_err_Smag = np.abs(S_mag - S_mag0) / (np.abs(S_mag0) )

plt.figure(figsize=(10,6))
plt.plot(tau, rel_err_Smag, label=r'Relative Error $|\vec{S}_1+\vec{S}_2|$')
plt.yscale('log')
plt.xlim(tau[0], tau[-1])
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel('Relative Error', fontsize=18)
plt.title(r'Relative Error of $|\vec{S}_1+\vec{S}_2|$ vs Time', fontsize=18)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Extract the angular momentum vector and check the relative error 
L_vectors = np.array([Lx_sol, Ly_sol, Lz_sol]).T  # shape (N, 3)
angular_momentum_magnitude = np.linalg.norm(L_vectors, axis=1)
L0 = angular_momentum_magnitude[0]

# Comparison of individual terms - S_mag DOES NOT have to conserve
plt.figure(figsize=(10,6))
plt.plot(tau, angular_momentum_magnitude, label=r'$|\vec{\mathcal{L}}|$', color='blue', linewidth=2)
plt.plot(tau, S_mag, label=r'$|\vec{\mathcal{S}}_1 + \vec{\mathcal{S}}_2|$', color='red', linewidth=2)
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel('Magnitude', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()



# Plot the orbital angular momentum components over time
plt.figure(figsize=(10, 6))
plt.plot(tau, Lx_sol, label=r'Regularized $L_x$', color='blue')
plt.plot(tau, Ly_sol, label=r'Regularized $L_y$', color='green')
plt.plot(tau, Lz_sol, label=r'Regularized $L_z$', color='red')
plt.xlim(tau[0], tau[-1])
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel('Angular Momentum Components', fontsize=18)
plt.title('Regularized Angular Momentum Components vs Time', fontsize=18)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


# Extract the Hamiltonian at each time step
H_sol = np.array([
    Hamiltonian(
        r_sol[i], pr_sol[i], Lx_sol[i], Ly_sol[i], Lz_sol[i],
        S1x_sol[i], S1y_sol[i], S1z_sol[i],
        S2x_sol[i], S2y_sol[i], S2z_sol[i]
    )
    for i in range(len(tau))
])
H0 = H_sol[0]
relative_energy_change = np.abs(H_sol - H0) / np.abs(H0)
print("Max relative energy change:", np.max(relative_energy_change))
rel_H_err = np.abs(H_sol - H0) / np.abs(H0)
print("Relative orbital angular momentum variation:", np.max(np.abs(angular_momentum_magnitude - L0) / L0))

# Define the factor of LS product
S_vectors = S1_vectors + S2_vectors  # shape (N, 3)
dot_LS = np.sum(L_vectors * S_vectors, axis=1)
L_norm = np.linalg.norm(L_vectors, axis=1)
S_norm = np.linalg.norm(S_vectors, axis=1)
angle_LS = np.arccos(np.clip(dot_LS / (L_norm * S_norm + 1e-14), -1.0, 1.0))

plt.figure(figsize=(10,6))
plt.plot(tau, angle_LS, label=r'$|\vec{\mathcal{L}}|$', color='blue', linewidth=2)
plt.plot(tau, S_mag, label='S_mag', color='red', linewidth=2)
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel('Magnitude', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()


#ANGULAR VELOCITY (z-component of Omega_L) - from the spherical defintion Lz = pphi, thus, emplyoing the Hamiltonian canonical equation w = dphi/dt = \partial H/dLz

OmegaL_newtonian_z = np.array([
    Lz_sol[i] / (G**2 * M**2 * r_sol[i]**2 * mu)
    for i in range(len(tau))
])
OmegaL_1PN_z = np.array([
    (Lz_sol[i] * (-2*(3 + nu) + r_sol[i] * (pr_sol[i]**2 + (Lx_sol[i]**2 + Ly_sol[i]**2 + Lz_sol[i]**2) / (G**2 * M**2 * r_sol[i]**2 * mu**2)) * (-1 + 3*nu)))
    / (2 * c**2 * G**2 * M**2 * r_sol[i]**3 * mu)
    for i in range(len(tau))
])
OmegaL_15PN_z = np.array([
    (2 * (S1z_sol[i] + (3*m2*S1z_sol[i])/(4*m1) + S2z_sol[i] + (3*m1*S2z_sol[i])/(4*m2)))
    / (c**2 * G**2 * M**3 * r_sol[i]**3)
    for i in range(len(tau))
])
OmegaL_2PN_z = np.array([
    (1 / (8 * c**4 * G**6 * M**6 * r_sol[i]**6 * mu**5)) * (
        Lz_sol[i] * (
            -4 * G**4 * M**4 * pr_sol[i]**2 * r_sol[i]**3 * mu**4 * nu**2
            + 8 * G**4 * M**4 * r_sol[i]**2 * mu**4 * (5 + 8*nu)
            + 3 * (Lx_sol[i]**2 + Ly_sol[i]**2 + Lz_sol[i]**2 + G**2 * M**2 * pr_sol[i]**2 * r_sol[i]**2 * mu**2)**2 * (1 + 5*(-1 + nu)*nu)
            + 4 * G**2 * M**2 * r_sol[i] * mu**2 * (Lx_sol[i]**2 + Ly_sol[i]**2 + Lz_sol[i]**2 + G**2 * M**2 * pr_sol[i]**2 * r_sol[i]**2 * mu**2) * (5 - nu*(20 + 3*nu))
        )
    )
    for i in range(len(tau))
])
OmegaL_S1S1_z = np.array([
    (3 * m2 * (
        (-Lz_sol[i]) * (Lx_sol[i] * S1x_sol[i] + Ly_sol[i] * S1y_sol[i]) +
        (Lx_sol[i]**2 + Ly_sol[i]**2) * S1z_sol[i]
    ) * (Lx_sol[i] * S1x_sol[i] + Ly_sol[i] * S1y_sol[i] + Lz_sol[i] * S1z_sol[i]))
    / (c**2 * G**2 * (Lx_sol[i]**2 + Ly_sol[i]**2 + Lz_sol[i]**2)**2 * M**3 * m1 * r_sol[i]**3)
    for i in range(len(tau))
])
OmegaL_S2S2_z = np.array([
    (3 * m1 * (
        (-Lz_sol[i]) * (Lx_sol[i] * S2x_sol[i] + Ly_sol[i] * S2y_sol[i]) +
        (Lx_sol[i]**2 + Ly_sol[i]**2) * S2z_sol[i]
    ) * (Lx_sol[i] * S2x_sol[i] + Ly_sol[i] * S2y_sol[i] + Lz_sol[i] * S2z_sol[i]))
    / (c**2 * G**2 * (Lx_sol[i]**2 + Ly_sol[i]**2 + Lz_sol[i]**2)**2 * M**3 * m2 * r_sol[i]**3)
    for i in range(len(tau))
])
OmegaL_S1S2_z = np.array([
    (3 * (
        -2 * Lz_sol[i] * (Lx_sol[i] * S1x_sol[i] + Ly_sol[i] * S1y_sol[i]) +
        (Lx_sol[i]**2 + Ly_sol[i]**2 - Lz_sol[i]**2) * S1z_sol[i]
    ) * (Lx_sol[i] * S2x_sol[i] + Ly_sol[i] * S2y_sol[i]) +
    3 * (
        (Lx_sol[i]**2 + Ly_sol[i]**2 - Lz_sol[i]**2) * (Lx_sol[i] * S1x_sol[i] + Ly_sol[i] * S1y_sol[i]) +
        2 * (Lx_sol[i]**2 + Ly_sol[i]**2) * Lz_sol[i] * S1z_sol[i]
    ) * S2z_sol[i]
    ) / (
        c**2 * G**2 * (Lx_sol[i]**2 + Ly_sol[i]**2 + Lz_sol[i]**2)**2 * M**3 * r_sol[i]**3
    )
    for i in range(len(tau))
])

# ANGULAR VELOCITY EXTRACTION
ang_vel = []
for i in range(len(tau)):
    # Recompute Omega_L at each time step using solution arrays
    Omega_Lz = (
        OmegaL_newtonian_z[i] + OmegaL_1PN_z[i] + OmegaL_15PN_z[i] +
        OmegaL_2PN_z[i] + OmegaL_S1S1_z[i] + OmegaL_S2S2_z[i] + OmegaL_S1S2_z[i]
    )
    ang_vel.append(Omega_Lz)
ang_vel = np.array(ang_vel)

plt.figure(figsize=(10,6))
plt.plot(tau, ang_vel, label=r'$\omega = \Omega_L^z$')
plt.xlabel('Time ($\tau$)', fontsize=18)
plt.ylabel(r'Orbital Angular Velocity $\omega$', fontsize=18)
plt.title(r'Evolution of Orbital Angular Velocity $\omega = \Omega_L^z$', fontsize=18)
plt.legend()
plt.grid(True)
plt.show()

# Integrate angular velocity to get phase (employing the cumulative_trapezoid SciPy library)
orbital_phase = cumulative_trapezoid(ang_vel, tau, initial=0)

# Plot the orbital phase vs time
plt.figure(figsize=(10,6))
plt.plot(tau, orbital_phase, label='Orbital Phase')
plt.xlabel('Time ($\\tau$)', fontsize=18)
plt.ylabel('Orbital Phase $\\Phi$', fontsize=18)
plt.title('Orbital Phase vs Time', fontsize=18)
plt.legend()
plt.grid(True)
plt.show()

#ORBITAL VELOCITY magnitude

v_orb = np.sqrt(pr_sol**2 + (Lx_sol**2 + Ly_sol**2 + Lz_sol**2) / (G**2 * M**2 * r_sol**2 * mu**2))

# v_orb vs r
plt.figure(figsize=(8,5))
plt.plot(r_sol, v_orb, lw=1)
plt.xlabel(r'$r$', fontsize=18)
plt.ylabel(r'$v_{\rm orb}$', fontsize=18)
plt.title(r'Orbital Velocity $v_{\rm orb}$ vs Radius $r$', fontsize=18)
plt.grid(True)
plt.show()

# v_orb vs tau
plt.figure(figsize=(8,5))
plt.plot(tau, v_orb, lw=1)
plt.xlabel(r'Time ($\tau$)', fontsize=18)
plt.ylabel(r'$v_{\rm orb}$', fontsize=18)
plt.title(r'Orbital Velocity $v_{\rm orb}$ vs Time', fontsize=18)
plt.grid(True)
plt.show()

# Angle between L and S = S1 + S2
dot_LS = np.sum(L_vectors * S_vectors, axis=1)
L_norm = np.linalg.norm(L_vectors, axis=1)
S_norm = np.linalg.norm(S_vectors, axis=1)
angle_LS = np.arccos(np.clip(dot_LS / (L_norm * S_norm + 1e-14), -1.0, 1.0))

# Angle between S1 and S2
dot_S1S2 = np.sum(S1_vectors * S2_vectors, axis=1)
S1_norm = np.linalg.norm(S1_vectors, axis=1)
S2_norm = np.linalg.norm(S2_vectors, axis=1)
angle_S1S2 = np.arccos(np.clip(dot_S1S2 / (S1_norm * S2_norm + 1e-14), -1.0, 1.0))

# Angle between L and S1
dot_LS1 = np.sum(L_vectors * S1_vectors, axis=1)
angle_LS1 = np.arccos(np.clip(dot_LS1 / (L_norm * S1_norm + 1e-14), -1.0, 1.0))

# Angle between L and S2
dot_LS2 = np.sum(L_vectors * S2_vectors, axis=1)
angle_LS2 = np.arccos(np.clip(dot_LS2 / (L_norm * S2_norm + 1e-14), -1.0, 1.0))

# Convert to degrees
angle_LS_deg = np.degrees(angle_LS)
angle_LS1_deg = np.degrees(angle_LS1)
angle_LS2_deg = np.degrees(angle_LS2)
angle_S1S2_deg = np.degrees(angle_S1S2)

plt.figure(figsize=(10,6))
plt.plot(tau, angle_S1S2_deg, color='red', label=r'$\theta_{\vec{\mathcal{S}}_1 \cdot \vec{\mathcal{S}}_2}$')
plt.plot(tau, angle_LS_deg, color='purple', label=r'$\theta_{\vec{\mathcal{L}} \cdot \vec{\mathcal{S}}}$')
#plt.plot(tau, angle_LS1_deg, color='green', label=r'$\theta_{\vec{\mathcal{L}} \cdot \vec{\mathcal{S}}_1}$')
#plt.plot(tau, angle_LS2_deg, color='orange', label=r'$\theta_{\vec{\mathcal{L}} \cdot \vec{\mathcal{S}}_2}$')
plt.xlim(tau[0], tau[-1])
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel(r'$\theta\ (^{\circ})$', fontsize=18)
plt.legend(fontsize=18, loc='best')
plt.grid(True)
plt.show()

# 3D SPIN-ANGULAR MOMENTUM TRAJECTORY PLOTS
fig = plt.figure(figsize=(10, 7), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
# Set the background color of the 3D plot to white
# Set 3D pane background to white
ax.set_facecolor('white')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


# Plot the trajectories
ax.plot(L_vectors[:,0], L_vectors[:,1], L_vectors[:,2], label=r'Trajectory of $\vec{\mathcal{L}}$', color='blue')
ax.plot(S1_vectors[:,0], S1_vectors[:,1], S1_vectors[:,2], label=r'Trajectory of $\vec{\mathcal{S}}_1$', color='green')
ax.plot(S2_vectors[:,0], S2_vectors[:,1], S2_vectors[:,2], label=r'Trajectory of $\vec{\mathcal{S}}_2$', color='orange')

# Choose an index along the trajectory for the arrow (initial direction)
arrow_scale = 0.025
i_arrow = 0  # Set as the initial vector orientation

# L arrow
L_base = L_vectors[i_arrow]
L_dir = L_vectors[i_arrow+1] - L_vectors[i_arrow]
L_dir = L_dir / np.linalg.norm(L_dir) * arrow_scale
ax.quiver(L_base[0], L_base[1], L_base[2],
          L_dir[0], L_dir[1], L_dir[2],
          color='navy', linewidth=2.5, arrow_length_ratio=0.85, label=r'$\vec{\mathcal{L}}_0$')

# S1 arrow
S1_base = S1_vectors[i_arrow]
S1_dir = S1_vectors[i_arrow+1] - S1_vectors[i_arrow]
S1_dir = S1_dir / np.linalg.norm(S1_dir) * arrow_scale
ax.quiver(S1_base[0], S1_base[1], S1_base[2],
          S1_dir[0], S1_dir[1], S1_dir[2],
          color='purple', linewidth=2.5, arrow_length_ratio=0.85, label=r'$\vec{\mathcal{S}}_{1_0}$')

# S2 arrow
S2_base = S2_vectors[i_arrow]
S2_dir = S2_vectors[i_arrow+1] - S2_vectors[i_arrow]
S2_dir = S2_dir / np.linalg.norm(S2_dir) * arrow_scale
ax.quiver(S2_base[0], S2_base[1], S2_base[2],
          S2_dir[0], S2_dir[1], S2_dir[2],
          color='red', linewidth=2.5, arrow_length_ratio=0.85, label=r'$\vec{\mathcal{S}}_{2_0}$')
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$y$', fontsize=18)
ax.set_zlabel(r'$z$', fontsize=18)
plt.legend(fontsize=18, loc='upper right') #, bbox_to_anchor=(-0.1, 1.0))   #, bbox_to_anchor=(0.95, 1.0)
plt.tight_layout()
plt.show()


N = min(len(r_sol), len(pr_sol))
plt.plot(r_sol[:N], pr_sol[:N], lw=1)
plt.xlabel(r'$r$', fontsize=18)
plt.ylabel(r'$p_r$', fontsize=18)
plt.title(r'Radial Momentum $p_r$ vs Radius $r$', fontsize=18)
plt.grid(True)
plt.show()



# PLOT THETA_{LS1}, THETA_{LS2}, THETA_{LS} AND THETA_{S1S2} for different chi1 values for a fixed chi2 value

chi1_values = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]  # Example spin parameters for body 1
chi2 = -1.0  # Fixed spin parameter for body 2


# Store results for each chi1


for chi1 in chi1_values:
    
    initial_state = [r_0, pr_0, Lx0, Ly0, Lz0, S1x0, S1y0, S1z0, S2x0, S2y0, S2z0]
    sol = solve_ivp(
        equations_of_motion,
        t_span,
        initial_state,
        method='DOP853',
        atol=1e-18,
        rtol=2.23e-14,
        t_eval=tau
    )
    r_sol, pr_sol, Lx_sol, Ly_sol, Lz_sol, S1x_sol, S1y_sol, S1z_sol, S2x_sol, S2y_sol, S2z_sol = sol.y


# Store values as a list for a plot including all chi1 values
angle_S1S2_deg_list = []
angle_LS_deg_list = []
angle_LS1_deg_list = []
angle_LS2_deg_list = []

for chi1 in chi1_values:
    # Update spin parameters for each chi1
    S1m = chi1 * G * m1**2 / c
    S2m = chi2 * G * m2**2 / c
    S1x0 = S1m * 2/3
    S1y0 = -S1m * 1/3
    S1z0 = -S1m * 2/3
    S2x0 = -S2m * 2/7
    S2y0 = -S2m * 3/7
    S2z0 = S2m * 6/7
    initial_state = [r_0, pr_0, Lx0, Ly0, Lz0, S1x0, S1y0, S1z0, S2x0, S2y0, S2z0]
    sol = solve_ivp(
        equations_of_motion,
        t_span,
        initial_state,
        method='DOP853',
        atol=1e-18,
        rtol=2.23e-14,
        t_eval=tau
    )
    r_sol, pr_sol, Lx_sol, Ly_sol, Lz_sol, S1x_sol, S1y_sol, S1z_sol, S2x_sol, S2y_sol, S2z_sol = sol.y
    L_vectors = np.array([Lx_sol, Ly_sol, Lz_sol]).T
    S1_vectors = np.array([[S1x_sol[i], S1y_sol[i], S1z_sol[i]] for i in range(len(tau))])
    S2_vectors = np.array([[S2x_sol[i], S2y_sol[i], S2z_sol[i]] for i in range(len(tau))])
    S1_mag = np.linalg.norm(S1_vectors, axis=1)
    S2_mag = np.linalg.norm(S2_vectors, axis=1)
    S1_vectors = S1_vectors * (S1m / S1_mag[:, np.newaxis])
    S2_vectors = S2_vectors * (S2m / S2_mag[:, np.newaxis])
    S_vectors = S1_vectors + S2_vectors
    L_norm = np.linalg.norm(L_vectors, axis=1)
    S_norm = np.linalg.norm(S_vectors, axis=1)
    S1_norm = np.linalg.norm(S1_vectors, axis=1)
    S2_norm = np.linalg.norm(S2_vectors, axis=1)
    dot_LS = np.sum(L_vectors * S_vectors, axis=1)
    dot_S1S2 = np.sum(S1_vectors * S2_vectors, axis=1)
    dot_LS1 = np.sum(L_vectors * S1_vectors, axis=1)
    dot_LS2 = np.sum(L_vectors * S2_vectors, axis=1)
    angle_LS = np.arccos(np.clip(dot_LS / (L_norm * S_norm + 1e-14), -1.0, 1.0))
    angle_S1S2 = np.arccos(np.clip(dot_S1S2 / (S1_norm * S2_norm + 1e-14), -1.0, 1.0))
    angle_LS1 = np.arccos(np.clip(dot_LS1 / (L_norm * S1_norm + 1e-14), -1.0, 1.0))
    angle_LS2 = np.arccos(np.clip(dot_LS2 / (L_norm * S2_norm + 1e-14), -1.0, 1.0))
    angle_LS_deg_list.append(np.degrees(angle_LS))
    angle_S1S2_deg_list.append(np.degrees(angle_S1S2))
    angle_LS1_deg_list.append(np.degrees(angle_LS1))
    angle_LS2_deg_list.append(np.degrees(angle_LS2))

#plt.figure(figsize=(14,6))
import matplotlib.cm as cm
# Prepare colors
num_curves = len(chi1_values) * 2  # You use two lines per chi1
colors = cm.get_cmap('tab20', num_curves)

plt.figure(figsize=(14, 6))
for i, chi1 in enumerate(chi1_values):
    color_ls1 = colors(i * 2)
    plt.plot(tau, angle_LS1_deg_list[i],
             label=fr'$\chi_1={chi1}$ $(\vec{{\mathcal{{L}}}} \cdot \vec{{\mathcal{{S}}}}_1)$', linewidth=2.5, color=color_ls1)
    color_ls2 = colors(i * 2 + 1)
    plt.plot(tau, angle_LS2_deg_list[i],
             label=fr'$\chi_1={chi1}$ $(\vec{{\mathcal{{L}}}} \cdot \vec{{\mathcal{{S}}}}_2)$',
             linewidth=2.5, color=color_ls2)
#linestyle='-.'

plt.xlim(tau[0], tau[-1]/20)
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel(r'$\theta\ (^{\circ})$', fontsize=18)
#plt.title(r'Angle between $\vec{\mathcal{L}}$ and $\vec{\mathcal{S}}_1$', fontsize=16)
plt.legend(fontsize=9, bbox_to_anchor=(1.0, 0.5), loc='center left')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
import matplotlib.cm as cm
num_curves = len(chi1_values) * 2  # (L·S1) and (L·S2) for each chi1
colors = cm.get_cmap('tab20', num_curves)
for i, chi1 in enumerate(chi1_values):
    color_ls = colors(i * 2)
    color_s1s2 = colors(i * 2 + 1)
    plt.plot(tau, angle_LS_deg_list[i], label=fr'$\chi_1={chi1}$ $(\vec{{\mathcal{{L}}}} \cdot \vec{{\mathcal{{S}}}})$', linewidth=2.5, color=color_ls)
    plt.plot(tau, angle_S1S2_deg_list[i], label=fr'$\chi_1={chi1}$ $(\vec{{\mathcal{{S}}}}_1 \cdot \vec{{\mathcal{{S}}}}_2)$', linewidth=2.5, color=color_s1s2)
plt.xlim(tau[0], tau[-1]/20)
plt.xlabel(r'$t$ (M)', fontsize=18)
plt.ylabel(r'$\theta\ (^{\circ})$', fontsize=18)
plt.legend(fontsize=9, bbox_to_anchor=(1.0, 0.5), loc='center left')
plt.grid(True)
plt.show()
