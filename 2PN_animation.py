# ANIMATION OF A BINARY BLACK HOLE SYSTEM UP TO THE SECOND POST-NEWTONIAN ORDER
# Import of the necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator, FuncFormatter
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from joblib import Parallel, delayed

mpl.rcParams['agg.path.chunksize'] = 100000   # For enhanced plotting

np.set_printoptions(precision=15)

# Constants (Geometrized unit system)
G = 1.0             # Gravitational constant
c = 1.0             # Speed of light

# Black hole masses (geometrized units)
m1 = 0.5            # Mass of black hole 1
m2 = 0.5            # Mass of black hole 2

# Derived quantities
M = m1 + m2         # Total mass of the system
mu = m1 * m2 / M    # Reduced mass
nu = mu / M         # Symmetric mass ratio
GM = G * M          # Gravitational parameter

# Spin parameters
chi1 = -1.0         # Dimensionless spin parameter for the first black hole
chi2 = 1.0          # Dimensionless spin parameter for the second black hole

# Effective spin constants
sigma1 = 1.0 + 3 * m2 / (4 * m1)
sigma2 = 1.0 + 3 * m1 / (4 * m2)

# Spin magnitudes
S1m = chi1 * G * m1**2 / c
S2m = chi2 * G * m2**2 / c

# Initial position of each black hole centre (in units of M)
x01 = 15.0
y01 = 0.0
z01 = 5.0
x02 = - 15.0
y02 = 0.0
z02 = -5.0

# Initial axis separation
xdiff = x01 - x02
ydiff = y01 - y02
zdiff = z01 - z02

# Initial centre of mass position
r_0x = xdiff
r_0y = ydiff
r_0z = zdiff
r_sep = np.sqrt(r_0x**2 + r_0y**2 + r_0z**2)
r_comx = (m1 * x01 + m2 * x02)/( M)
r_comy = (m1 * y01 + m2 * y02)/( M)
r_comz = (m1 * z01 + m2 * z02)/( M)

# Initial conditions for the system's centre of mass
r_0 = r_sep                         # Initial distance between the two black holes
phi_0 = np.arctan2(r_0y, r_0x)      # Initial azimuthal angle
theta_0 = np.arccos(r_0z/r_0)       # Initial polar angle
pr_0 = 2e-16                        # Initial radial momentum (very small to avoid purely circular orbit) 
pphi_0 = 0.6 * np.sqrt(r_0)         # Initial azimuthal momentum
ptheta_0 = 0.4 * np.sqrt(r_0)       # Initial polar momentum


S1x0 = 2/3 * S1m                    # Initial x-component of spin vector for black hole 1
S1y0 = -1/3 * S1m                   # Initial y-component of spin vector for black hole 1
S1z0 = -2/3 * S1m                   # Initial z-component of spin vector for black hole 1

S2x0 = -2/7 * S2m                   # Initial x-component of spin vector for black hole 2
S2y0 = -3/7 * S2m                   # Initial y-component of spin vector for black hole 2
S2z0 = 6/7 * S2m                    # Initial z-component of spin vector for black hole 2


def cot(x):
    return np.cos(x) / np.sin(x)

def csc(x):
    return 1 / np.sin(x)

def spin_vectors(S1x, S1y, S1z, S2x, S2y, S2z):

    S1 = np.array([S1x, S1y, S1z])
    S2 = np.array([S2x, S2y, S2z])

    return S1, S2


def angular_momentum(phi, theta, pphi, ptheta):
    
    L = GM * mu * np.array([ -pphi*np.cos(phi)*cot(theta) - ptheta*np.sin(phi), 
        ptheta*np.cos(phi) - pphi*cot(theta)*np.sin(phi), 
        pphi])
    
    return L

def angular_momentum_magnitude(L):
    return np.linalg.norm(L)

def total_angular_momentum(phi, theta, pphi, ptheta, S1x, S1y, S1z, S2x, S2y, S2z):
    return GM * mu * np.array([ -pphi*np.cos(phi)*cot(theta) - ptheta*np.sin(phi), 
        ptheta*np.cos(phi) - pphi*cot(theta)*np.sin(phi), 
        pphi]) + np.array([S1x, S1y, S1z]) + np.array([S2x, S2y, S2z])


# Defining the Hamiltonian dynamics up to the second post-Newtonian order
def HN(r, theta, pr, pphi, ptheta):
    
    return (mu*(ptheta**2 + r*(-2 + pr**2*r) + pphi**2*csc(theta)**2))/(2*r**2)
    
def H1PN(r, theta, pr, pphi, ptheta):
    return (1/(8*c**2*r**4))*(mu*(4*r**2 + (-1 + 3*nu)*(ptheta**2 + pr**2*r**2 + pphi**2*csc(theta)**2)**2 - 
    4*r*(ptheta**2*(3 + nu) + pr**2*r**2*(3 + 2*nu) + pphi**2*(3 + nu)*csc(theta)**2)))

def H1_5PN(r, phi, theta, ptheta, pphi, S1x, S1y, S1z, S2x, S2y, S2z):
    return (1/(2*c**2*G*M**2*m1*m2*r**3))*(mu*(pphi*(3*m2**2*S1z + 3*m1**2*S2z + 4*m1*m2*(S1z + S2z)) - 
    (3*m2**2*S1x + 3*m1**2*S2x + 4*m1*m2*(S1x + S2x))*(pphi*np.cos(phi)*cot(theta) + 
      ptheta*np.sin(phi)) + (3*m2**2*S1y + 3*m1**2*S2y + 4*m1*m2*(S1y + S2y))*
     (ptheta*np.cos(phi) - pphi*cot(theta)*np.sin(phi))))
  
def H2PN(r, theta, pr, ptheta, pphi):
    
    return (1/(16*c**4*r**6))*(mu*(24*pr**2*r**4*nu - 6*pr**4*r**5*nu**2 - 4*r**3*(1 + 3*nu) - 
    4*pr**2*r**3*nu**2*(ptheta**2 + pr**2*r**2 + pphi**2*csc(theta)**2) + 
    8*r**2*(5 + 8*nu)*(ptheta**2 + pr**2*r**2 + pphi**2*csc(theta)**2) - 
    2*r*(-5 + nu*(20 + 3*nu))*(ptheta**2 + pr**2*r**2 + pphi**2*csc(theta)**2)**2 + 
    (1 + 5*(-1 + nu)*nu)*(ptheta**2 + pr**2*r**2 + pphi**2*csc(theta)**2)**3))
    

def HS1S1(r, phi, theta, S1x, S1y, S1z):

    return (1/(8*c**2*G**2*M**3*m1*r**3))*(m2*(-((S1x**2 + S1y**2 - 2*S1z**2)*(1 + 3*np.cos(2*theta))) + 
    6*(S1x - S1y)*(S1x + S1y)*np.cos(2*phi)*np.sin(theta)**2 + 12*S1x*S1z*np.cos(phi)*np.sin(2*theta) + 
    12*S1y*S1z*np.sin(2*theta)*np.sin(phi) + 12*S1x*S1y*np.sin(theta)**2*np.sin(2*phi)))


def HS2S2(r, phi, theta, S2x, S2y, S2z):

    return (1/(8*c**2*G**2*M**3*m2*r**3))*(m1*(-((S2x**2 + S2y**2 - 2*S2z**2)*(1 + 3*np.cos(2*theta))) + 
    6*(S2x - S2y)*(S2x + S2y)*np.cos(2*phi)*np.sin(theta)**2 + 12*S2x*S2z*np.cos(phi)*np.sin(2*theta) + 
    12*S2y*S2z*np.sin(2*theta)*np.sin(phi) + 12*S2x*S2y*np.sin(theta)**2*np.sin(2*phi)))
    
    
def HS1S2(r, phi, theta, S1x, S1y, S1z, S2x, S2y, S2z):

    return -((1/(c**2*G**2*M**3*r**3))*(S1x*S2x + S1y*S2y + S1z*S2z - 
    3*(S1z*np.cos(theta) + np.sin(theta)*(S1x*np.cos(phi) + S1y*np.sin(phi)))*
     (S2z*np.cos(theta) + np.sin(theta)*(S2x*np.cos(phi) + S2y*np.sin(phi)))))


def Hamiltonian(r, phi, theta, pr, pphi, ptheta, S1x, S1y, S1z, S2x, S2y, S2z):
    return (HN(r, theta, pr, pphi, ptheta)
        + H1PN(r, theta, pr, pphi, ptheta)
        + H1_5PN(r, phi, theta, ptheta, pphi, S1x, S1y, S1z, S2x, S2y, S2z)
        + H2PN(r, theta, pr, ptheta, pphi)
        + HS1S1(r, phi, theta, S1x, S1y, S1z)
        + HS2S2(r, phi, theta, S2x, S2y, S2z)
        + HS1S2(r, phi, theta, S1x, S1y, S1z, S2x, S2y, S2z))


# Equations (EoM) that govern the motion of the binary black hole system's centre of mass
def equations_of_motion(tau, state):
    r, phi, theta, pr, pphi, ptheta, S1x, S1y, S1z, S2x, S2y, S2z = state
    

    #DRDTAU TERMS
    
    drdtau_newtonian = mu * pr

    drdtau_1PN = (mu*pr*(-2*(3 + 2*nu)*r + (-1 + 3*nu)*(ptheta**2 + csc(theta)**2*pphi**2 + 
      pr**2*r**2)))/(2*c**2*r**2)

    drdtau_15PN = 0

    drdtau_2PN = (1/(8*c**4*r**4))*(mu*pr*(24*nu*r**2 + 8*(5 + 8*nu)*r**2 - 
    16*nu**2*pr**2*r**3 - 4*nu**2*r*(ptheta**2 + csc(theta)**2*pphi**2 + 
      pr**2*r**2) - 4*(-5 + nu*(20 + 3*nu))*r*(ptheta**2 + csc(theta)**2*pphi**2 + 
      pr**2*r**2) + 3*(1 + 5*(-1 + nu)*nu)*(ptheta**2 + csc(theta)**2*pphi**2 + 
       pr**2*r**2)**2))
   
    drdtau_S1S1 = 0

    drdtau_S2S2 = 0

    drdtau_S1S2 = 0


    #DPHIDTAU TERMS

    dphidtau_newtonian = (mu*csc(theta)**2*pphi)/r**2

    dphidtau_1PN = (mu*csc(theta)**2*pphi*(-2*(3 + nu)*r + 
    (-1 + 3*nu)*(ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2)))/(2*c**2*r**4)

    dphidtau_15PN = (1/(2*c**2*G*M**2*m1*m2*r**3))*(mu*(3*m2**2*S1z + 3*m1**2*S2z + 4*m1*m2*(S1z + S2z) - 
    (3*m2**2*S1x + 3*m1**2*S2x + 4*m1*m2*(S1x + S2x))*np.cos(phi)*cot(theta) - 
    (3*m2**2*S1y + 3*m1**2*S2y + 4*m1*m2*(S1y + S2y))*cot(theta)*np.sin(phi)))
    
    dphidtau_2PN = (1/(8*c**4*r**6))*(mu*csc(theta)**2*pphi*(8*(5 + 8*nu)*r**2 - 
    4*nu**2*pr**2*r**3 - 4*(-5 + nu*(20 + 3*nu))*r*
     (ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2) + 
    3*(1 + 5*(-1 + nu)*nu)*(ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2)**2))

    dphidtau_S1S1 = 0

    dphidtau_S2S2 = 0

    dphidtau_S1S2 = 0


    #DTHETADTAU TERMS


    dthetadtau_newtonian = (mu*ptheta)/r**2

    dthetadtau_1PN = (mu*(-8*(3 + nu)*ptheta*r + 4*(-1 + 3*nu)*ptheta*(ptheta**2 + csc(theta)**2*pphi**2 + 
      pr**2*r**2)))/(8*c**2*r**4)

    dthetadtau_15PN = (1/(2*c**2*G*M**2*m1*m2*r**3))*(mu*((3*m2**2*S1y + 3*m1**2*S2y + 4*m1*m2*(S1y + S2y))*
     np.cos(phi) - (3*m2**2*S1x + 3*m1**2*S2x + 4*m1*m2*(S1x + S2x))*np.sin(phi)))
    
    dthetadtau_2PN = (1/(8*c**4*r**6))*(mu*ptheta*(8*(5 + 8*nu)*r**2 - 4*nu**2*pr**2*r**3 - 
    4*(-5 + nu*(20 + 3*nu))*r*(ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2) + 
    3*(1 + 5*(-1 + nu)*nu)*(ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2)**2))
    
    dthetadtau_S1S1 = 0

    dthetadtau_S2S2 = 0

    dthetadtau_S1S2 = 0


    #DPRDTAU TERMS

    dprdtau_newtonian = (mu*(ptheta**2 + csc(theta)**2*pphi**2 - r))/r**3

    dprdtau_1PN = (1/(2*c**2*r**5))*(mu*((-1 + 3*nu)*ptheta**4 + (-1 + 3*nu)*csc(theta)**4*pphi**4 + 
    r**2*(2 - (3 + 2*nu)*pr**2*r) + ptheta**2*r*
     (-3*(3 + nu) + (-1 + 3*nu)*pr**2*r) + csc(theta)**2*pphi**2*
     ((-2 + 6*nu)*ptheta**2 + r*(-3*(3 + nu) + (-1 + 3*nu)*pr**2*r))))
    
    dprdtau_15PN = (1/(2*c**2*G*M**2*m1*m2*r**4))*(3*mu*((3*m2**2*S1z + 3*m1**2*S2z + 4*m1*m2*(S1z + S2z))*
     pphi - (3*m2**2*S1x + 3*m1**2*S2x + 4*m1*m2*(S1x + S2x))*
     (np.cos(phi)*cot(theta)*pphi + ptheta*np.sin(phi)) + 
    (3*m2**2*S1y + 3*m1**2*S2y + 4*m1*m2*(S1y + S2y))*(np.cos(phi)*ptheta - 
      cot(theta)*pphi*np.sin(phi))))
    
    dprdtau_2PN = (1/(8*c**4*r**7))*(mu*(3*(1 + 5*(-1 + nu)*nu)*ptheta**6 + 
    3*(1 + 5*(-1 + nu)*nu)*csc(theta)**6*pphi**6 + 
    ptheta**4*r*(25 - 5*nu*(20 + 3*nu) + 6*(1 + 5*(-1 + nu)*nu)*pr**2*r) + 
    ptheta**2*r**2*(16*(5 + 8*nu) + 3*pr**2*r*(10 - 8*nu*(5 + nu) + 
        (1 + 5*(-1 + nu)*nu)*pr**2*r)) + csc(theta)**4*pphi**4*
     (9*(1 + 5*(-1 + nu)*nu)*ptheta**2 + r*(25 - 5*nu*(20 + 3*nu) + 
        6*(1 + 5*(-1 + nu)*nu)*pr**2*r)) + 
    r**3*(-6*(1 + 3*nu) + pr**2*r*(40 + 88*nu + (5 - 4*nu*(5 + 2*nu))*pr**2*
         r)) + csc(theta)**2*pphi**2*(9*(1 + 5*(-1 + nu)*nu)*ptheta**4 + 
      2*ptheta**2*r*(25 - 5*nu*(20 + 3*nu) + 6*(1 + 5*(-1 + nu)*nu)*pr**2*r) + 
      r**2*(16*(5 + 8*nu) + 3*pr**2*r*(10 - 8*nu*(5 + nu) + 
          (1 + 5*(-1 + nu)*nu)*pr**2*r)))))

    dprdtau_S1S1 = (1/(8*c**2*G**2*M**3*m1*r**4))*(3*m2*(-((S1x**2 + S1y**2 - 2*S1z**2)*(1 + 3*np.cos(2*theta))) + 
    6*(S1x - S1y)*(S1x + S1y)*np.cos(2*phi)*np.sin(theta)**2 + 
    12*S1x*S1z*np.cos(phi)*np.sin(2*theta) + 12*S1y*S1z*np.sin(2*theta)*np.sin(phi) + 
    12*S1x*S1y*np.sin(theta)**2*np.sin(2*phi)))
    
    dprdtau_S2S2 = (1/(8*c**2*G**2*M**3*m2*r**4))*(3*m1*(-((S2x**2 + S2y**2 - 2*S2z**2)*(1 + 3*np.cos(2*theta))) + 
    6*(S2x - S2y)*(S2x + S2y)*np.cos(2*phi)*np.sin(theta)**2 + 
    12*S2x*S2z*np.cos(phi)*np.sin(2*theta) + 12*S2y*S2z*np.sin(2*theta)*np.sin(phi) + 
    12*S2x*S2y*np.sin(theta)**2*np.sin(2*phi)))
    
    dprdtau_S1S2 = -((1/(c**2*G**2*M**3*r**4))*(3*(S1x*S2x + S1y*S2y + S1z*S2z - 
     3*(S1z*np.cos(theta) + np.sin(theta)*(S1x*np.cos(phi) + S1y*np.sin(phi)))*
      (S2z*np.cos(theta) + np.sin(theta)*(S2x*np.cos(phi) + S2y*np.sin(phi))))))
    

    #DPPHIDTAU TERMS

    dpphidtau_newtonian = 0

    dpphidtau_1PN = 0

    dpphidtau_15PN = -((1/(2*c**2*G*M**2*m1*m2*r**3))*(mu*((3*m2**2*S1y + 3*m1**2*S2y + 4*m1*m2*(S1y + S2y))*
      ((-np.cos(phi))*cot(theta)*pphi - ptheta*np.sin(phi)) - 
     (3*m2**2*S1x + 3*m1**2*S2x + 4*m1*m2*(S1x + S2x))*(np.cos(phi)*ptheta - 
       cot(theta)*pphi*np.sin(phi)))))
    
    dpphidtau_2PN = 0

    dpphidtau_S1S1 = (1/(c**2*G**2*M**3*m1*r**3))*(3*m2*np.sin(theta)*((-S1y)*np.cos(phi) + S1x*np.sin(phi))*
    (S1z*np.cos(theta) + np.sin(theta)*(S1x*np.cos(phi) + S1y*np.sin(phi))))
            
    dpphidtau_S2S2 = (1/(c**2*G**2*M**3*m2*r**3))*(3*m1*np.sin(theta)*((-S2y)*np.cos(phi) + S2x*np.sin(phi))*
    (S2z*np.cos(theta) + np.sin(theta)*(S2x*np.cos(phi) + S2y*np.sin(phi))))
            
    dpphidtau_S1S2 = -((1/(c**2*G**2*M**3*r**3))*(3*np.sin(theta)*
    (np.cos(theta)*((S1z*S2y + S1y*S2z)*np.cos(phi) - (S1z*S2x + S1x*S2z)*np.sin(phi)) + 
     np.sin(theta)*((S1y*S2x + S1x*S2y)*np.cos(2*phi) + ((-S1x)*S2x + S1y*S2y)*
        np.sin(2*phi)))))


    #DPTHETADTAU TERMS

    dpthetadtau_newtonian = (mu*cot(theta)*csc(theta)**2*pphi**2)/r**2

    dpthetadtau_1PN = -((1/(2*c**2*r**4))*(mu*cot(theta)*csc(theta)**2*pphi**2*
    (2*(3 + nu)*r - (-1 + 3*nu)*(ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2))))

    dpthetadtau_15PN = -((1/(2*c**2*G*M**2*m1*m2*r**3))*(mu*csc(theta)**2*pphi*
    ((3*m2**2*S1x + 3*m1**2*S2x + 4*m1*m2*(S1x + S2x))*np.cos(phi) + 
     (3*m2**2*S1y + 3*m1**2*S2y + 4*m1*m2*(S1y + S2y))*np.sin(phi))))

    dpthetadtau_2PN =  -((1/(8*c**4*r**6))*(mu*cot(theta)*csc(theta)**2*pphi**2*
    (-8*(5 + 8*nu)*r**2 + 4*nu**2*pr**2*r**3 + 4*(-5 + nu*(20 + 3*nu))*r*
      (ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2) - 
     3*(1 + 5*(-1 + nu)*nu)*(ptheta**2 + csc(theta)**2*pphi**2 + pr**2*r**2)**2)))

    dpthetadtau_S1S1 = -((1/(4*c**2*G**2*M**3*m1*r**3))*
    (3*m2*(4*S1z*np.cos(2*theta)*(S1x*np.cos(phi) + S1y*np.sin(phi)) + 
     np.sin(2*theta)*(S1x**2 + S1y**2 - 2*S1z**2 + (S1x - S1y)*(S1x + S1y)*np.cos(2*phi) + 
       2*S1x*S1y*np.sin(2*phi)))))
    
    dpthetadtau_S2S2 = -((1/(4*c**2*G**2*M**3*m2*r**3))*
    (3*m1*(4*S2z*np.cos(2*theta)*(S2x*np.cos(phi) + S2y*np.sin(phi)) + 
     np.sin(2*theta)*(S2x**2 + S2y**2 - 2*S2z**2 + (S2x - S2y)*(S2x + S2y)*np.cos(2*phi) + 
       2*S2x*S2y*np.sin(2*phi)))))
    
    dpthetadtau_S1S2 = (1/(c**2*G**2*M**3*r**3))*(-3*np.cos(2*theta)*((S1z*S2x + S1x*S2z)*np.cos(phi) + 
     (S1z*S2y + S1y*S2z)*np.sin(phi)) - (3/2)*np.sin(2*theta)*
    (S1x*S2x + S1y*S2y - 2*S1z*S2z + (S1x*S2x - S1y*S2y)*np.cos(2*phi) + 
     (S1y*S2x + S1x*S2y)*np.sin(2*phi)))
    
    #SPIN TERMS
    
    dS1dtau_newtonian = np.array([0, 0, 0])
    dS2dtau_newtonian = np.array([0, 0, 0])
    
    
    dS1dtau_1PN = np.array([0, 0, 0])
    dS2dtau_1PN = np.array([0, 0, 0])
    
    OmegaS1_15PN = np.array([-((2*(1 + (3*m2)/(4*m1))*mu*(pphi*np.cos(phi)*cot(theta) + ptheta*np.sin(phi)))/(c**2*G*M**2*r**3)), 
    (2*(1 + (3*m2)/(4*m1))*mu*(ptheta*np.cos(phi) - pphi*cot(theta)*np.sin(phi)))/(c**2*G*M**2*r**3), 
    (2*(1 + (3*m2)/(4*m1))*mu*pphi)/(c**2*G*M**2*r**3)])
    
    OmegaS2_15PN = np.array([-((2*(1 + (3*m1)/(4*m2))*mu*(pphi*np.cos(phi)*cot(theta) + ptheta*np.sin(phi)))/(c**2*G*M**2*r**3)), 
    (2*(1 + (3*m1)/(4*m2))*mu*(ptheta*np.cos(phi) - pphi*cot(theta)*np.sin(phi)))/(c**2*G*M**2*r**3), 
    (2*(1 + (3*m1)/(4*m2))*mu*pphi)/(c**2*G*M**2*r**3)])
    

    dS1dtau_15PN = np.cross(OmegaS1_15PN, np.array([S1x, S1y, S1z]))

    dS2dtau_15PN = np.cross(OmegaS2_15PN, np.array([S2x, S2y, S2z]))

    dS1dtau_2PN = np.array([0, 0, 0])
    
    dS2dtau_2PN = np.array([0, 0, 0])
    
    OmegaS1_S1S1 = np.array([(m2*(-S1x - 3*S1x*np.cos(2*theta) + 6*S1z*np.cos(phi)*np.sin(2*theta) + 
     6*np.sin(theta)**2*(S1x*np.cos(2*phi) + S1y*np.sin(2*phi))))/(4*c**2*G**2*M**3*m1*r**3), 
        -((m2*(S1y + 3*S1y*np.cos(2*theta) - 6*S1z*np.sin(2*theta)*np.sin(phi) + 
      6*np.sin(theta)**2*(S1y*np.cos(2*phi) - S1x*np.sin(2*phi))))/(4*c**2*G**2*M**3*m1*r**3)), 
        (m2*(S1z + 3*S1z*np.cos(2*theta) + 3*np.sin(2*theta)*(S1x*np.cos(phi) + S1y*np.sin(phi))))/
    (2*c**2*G**2*M**3*m1*r**3)])


    dS1dtau_S1S1 = np.cross(OmegaS1_S1S1, np.array([S1x, S1y, S1z]))
    
    dS2dtau_S1S1 = np.array([0, 0, 0])
    
    OmegaS2_S2S2 = np.array([(m1*(-S2x - 3*S2x*np.cos(2*theta) + 6*S2z*np.cos(phi)*np.sin(2*theta) + 
     6*np.sin(theta)**2*(S2x*np.cos(2*phi) + S2y*np.sin(2*phi))))/(4*c**2*G**2*M**3*m2*r**3), 
    -((m1*(S2y + 3*S2y*np.cos(2*theta) - 6*S2z*np.sin(2*theta)*np.sin(phi) + 
      6*np.sin(theta)**2*(S2y*np.cos(2*phi) - S2x*np.sin(2*phi))))/(4*c**2*G**2*M**3*m2*r**3)), 
    (m1*(S2z + 3*S2z*np.cos(2*theta) + 3*np.sin(2*theta)*(S2x*np.cos(phi) + S2y*np.sin(phi))))/
    (2*c**2*G**2*M**3*m2*r**3)])
    
    dS1dtau_S2S2 = np.array([0, 0, 0])
    
    dS2dtau_S2S2 = np.cross(OmegaS2_S2S2, np.array([S2x, S2y, S2z]))
    
    OmegaS1_S1S2 = np.array([-((S2x - 3*np.cos(phi)*np.sin(theta)*(S2z*np.cos(theta) + np.sin(theta)*(S2x*np.cos(phi) + S2y*np.sin(phi))))/
    (c**2*G**2*M**3*r**3)), 
    -((S2y - 3*np.sin(theta)*np.sin(phi)*(S2z*np.cos(theta) + np.sin(theta)*(S2x*np.cos(phi) + S2y*np.sin(phi))))/
    (c**2*G**2*M**3*r**3)), 
     -((S2z - 3*np.cos(theta)*(S2z*np.cos(theta) + np.sin(theta)*(S2x*np.cos(phi) + S2y*np.sin(phi))))/
    (c**2*G**2*M**3*r**3))])
    
    OmegaS2_S1S2 = np.array([-((S1x - 3*np.cos(phi)*np.sin(theta)*(S1z*np.cos(theta) + np.sin(theta)*(S1x*np.cos(phi) + S1y*np.sin(phi))))/
    (c**2*G**2*M**3*r**3)), 
    -((S1y - 3*np.sin(theta)*np.sin(phi)*(S1z*np.cos(theta) + np.sin(theta)*(S1x*np.cos(phi) + S1y*np.sin(phi))))/
    (c**2*G**2*M**3*r**3)), 
    -((S1z - 3*np.cos(theta)*(S1z*np.cos(theta) + np.sin(theta)*(S1x*np.cos(phi) + S1y*np.sin(phi))))/
    (c**2*G**2*M**3*r**3))])
    
    dS1dtau_S1S2 = np.cross(OmegaS1_S1S2, np.array([S1x, S1y, S1z]))

    dS2dtau_S1S2 = np.cross(OmegaS2_S1S2, np.array([S2x, S2y, S2z]))

    drdtau = drdtau_newtonian + drdtau_1PN + drdtau_15PN + drdtau_2PN + drdtau_S1S1 + drdtau_S2S2 + drdtau_S1S2

    dphidtau = dphidtau_newtonian + dphidtau_1PN + dphidtau_15PN + dphidtau_2PN + dphidtau_S1S1 + dphidtau_S2S2 + dphidtau_S1S2

    dthetadtau = dthetadtau_newtonian + dthetadtau_1PN + dthetadtau_15PN + dthetadtau_2PN + dthetadtau_S1S1 + dthetadtau_S2S2 + dthetadtau_S1S2

    dprdtau = dprdtau_newtonian + dprdtau_1PN + dprdtau_15PN + dprdtau_2PN + dprdtau_S1S1 + dprdtau_S2S2 + dprdtau_S1S2

    dpphidtau = dpphidtau_newtonian + dpphidtau_1PN + dpphidtau_15PN + dpphidtau_2PN + dpphidtau_S1S1 + dpphidtau_S2S2 + dpphidtau_S1S2

    dpthetadtau = dpthetadtau_newtonian + dpthetadtau_1PN + dpthetadtau_15PN + dpthetadtau_2PN + dpthetadtau_S1S1 + dpthetadtau_S2S2 + dpthetadtau_S1S2

    dS1dtau = dS1dtau_newtonian + dS1dtau_1PN + dS1dtau_15PN + dS1dtau_2PN + dS1dtau_S1S1 + dS1dtau_S2S2 + dS1dtau_S1S2

    dS2dtau = dS2dtau_newtonian + dS2dtau_1PN + dS2dtau_15PN + dS2dtau_2PN + dS2dtau_S1S1 + dS2dtau_S2S2 + dS2dtau_S1S2

    return [drdtau, dphidtau, dthetadtau, dprdtau, dpphidtau, dpthetadtau, dS1dtau[0], dS1dtau[1], dS1dtau[2], dS2dtau[0], dS2dtau[1], dS2dtau[2]]


# Define the initial state and time span
initial_state = [r_0, phi_0, theta_0, pr_0, pphi_0, ptheta_0, S1x0, S1y0, S1z0, S2x0, S2y0, S2z0]

Tp = 2 * np.pi * np.sqrt(r_0**3 * GM**2)    # Orbital period
t_span = (0, 100 * Tp)                      # Start and end times
tau = np.linspace(0, 100 * Tp, 1000000)     # Time points for evaluation

# Ensure tau is within t_span
tau = np.clip(tau, t_span[0], t_span[1])

# Solving the equations of motion using solve_ivp
solution = solve_ivp(
    equations_of_motion,
    t_span,
    initial_state,
    method='DOP853',
    atol=1e-16,
    rtol=3e-14,
    t_eval=tau
)
r_sol, phi_sol, theta_sol, pr_sol, pphi_sol, ptheta_sol, S1x_sol, S1y_sol, S1z_sol, S2x_sol, S2y_sol, S2z_sol = solution.y


# Transforming the solution to cartesian coordinates
x_sol = r_sol * np.sin(theta_sol) * np.cos(phi_sol)
y_sol = r_sol * np.sin(theta_sol) * np.sin(phi_sol)
z_sol = r_sol * np.cos(theta_sol)

# Position of both black holes in M units (in cartesian coordinates)
x1 = GM * (r_comx + m2 * x_sol / M)
x2 = GM * (r_comx - m1 * x_sol / M)
y1 = GM * (r_comy + m2 * y_sol / M)
y2 = GM * (r_comy - m1 * y_sol / M)
z1 = GM * (r_comz + m2 * z_sol / M)
z2 = GM * (r_comz - m1 * z_sol / M)


# Parameters for the animation

fps = 170                                   # Frames per second
delta_n = 250                               # Step size
num_frames = (len(x1) - 1) // delta_n       # Number of frames

# Calculate spin vectors for each frame
S1v_sol, S2v_sol = [], []
for i in range(len(S1x_sol)):
    S1v, S2v = spin_vectors(S1x_sol[i], S1y_sol[i], S1z_sol[i], S2x_sol[i], S2y_sol[i], S2z_sol[i])
    S1v_sol.append(S1v)
    S2v_sol.append(S2v)
S1v_sol = np.array(S1v_sol)
S2v_sol = np.array(S2v_sol)


# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial plot elements
traj1, = ax.plot([], [], [], lw=1, label="Trajectory of BH1")  # Trajectory for point 1
traj2, = ax.plot([], [], [], lw=1, label="Trajectory of BH2")  # Trajectory for point 2
# Black hole sizes based on their masses
bh1, = ax.plot([], [], [], 'ko', markersize=8 * m1, label="BH1")  # Black hole 1 size
bh2, = ax.plot([], [], [], 'ko', markersize=8 * m2, label="BH2")  # Black hole 2 size
spin1 = None
spin2 = None
plotTitle = ax.set_title("Trajectory of a binary black hole system up to 2PN order")

margin = 2  # Add a little margin for aesthetics

x_min = min(np.min(x1), np.min(x2)) - margin
x_max = max(np.max(x1), np.max(x2)) + margin
y_min = min(np.min(y1), np.min(y2)) - margin
y_max = max(np.max(y1), np.max(y2)) + margin
z_min = min(np.min(z1), np.min(z2)) - margin
z_max = max(np.max(z1), np.max(z2)) + margin

ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_zlim([z_min, z_max])
ax.set_xlabel('X (M)')
ax.set_ylabel('Y (M)')
ax.set_zlabel('Z (M)')

# Draw a single frame of the animation
def drawframe(s):
    global spin1, spin2
    """Draw the s-th frame of the animation"""
    i = delta_n * s + 1
    traj1.set_data(x1[:i], y1[:i])
    traj1.set_3d_properties(z1[:i])
    bh1.set_data([x1[i - 1]], [y1[i - 1]])
    bh1.set_3d_properties([z1[i - 1]])

    # Update trajectory 2
    traj2.set_data(x2[:i], y2[:i])
    traj2.set_3d_properties(z2[:i])
    bh2.set_data([x2[i - 1]], [y2[i - 1]])
    bh2.set_3d_properties([z2[i - 1]])

    # Remove previous spin vectors
    if spin1:
        spin1.remove()
    if spin2:
        spin2.remove()

    # Update spin vectors
    spin1 = ax.quiver(x1[i - 1], y1[i - 1], z1[i - 1], S1v_sol[i - 1, 0], S1v_sol[i - 1, 1], S1v_sol[i - 1, 2], color='r', length=5, normalize=True)
    spin2 = ax.quiver(x2[i - 1], y2[i - 1], z2[i - 1], S2v_sol[i - 1, 0], S2v_sol[i - 1, 1], S2v_sol[i - 1, 2], color='b', length=5, normalize=True)

    return traj1, traj2, bh1, bh2, spin1, spin2

# Create the animation
anim = FuncAnimation(fig, drawframe, frames=num_frames, interval=1000 // fps)

# Display the animation
plt.show()
