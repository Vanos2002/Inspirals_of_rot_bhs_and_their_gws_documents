#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <array>
using namespace std;

// Constants (geometrized units)
const double G = 1.0;               // Gravitational constant in geometrized units
const double c = 1.0;               // Speed of light in geometrized units    

const double M_sun = 1.98847e30;                // 1 Solar mass in kg
const double G_SI = 6.6743e-11;                 // Gravitational constant in m^3/kg/s^2
const double c_SI = 299792458;                  // Speed of light in m/s

// Masses
const double m1_solar = 1.0;                        // Mass of black hole 1 in Solar masses
const double m2_solar = 1.0;                        // Mass of black hole 2 in Solar masses
const double M_total_solar = m1_solar + m2_solar;   // Total mass of the system in Solar masses

const double m1 = m1_solar / M_total_solar;                 // Mass of black hole 1
const double m2 = m2_solar / M_total_solar;                 // Mass of black hole 2
const double M = m1 + m2;                                   // Total mass of the system, defined as M=1
const double mu = m1 * m2 / M;                              // Reduced mass in code units   
const double nu = mu / M;                                   // Symmetric mass ratio in code units
const double Mc = pow(m1 * m2, 0.6) * pow(M, -0.2);         // Chirp mass in code units
const double M_kg = M_total_solar * M_sun;                  // Total mass in kg

const double time_unit_seconds = G_SI * M_kg / (c_SI * c_SI * c_SI);  // 1 code unit in seconds
const double sep_unit_meters = G_SI * M_kg / (c_SI * c_SI);           // 1 code unit in meters

// Initial separation scaling parameter
const double rsc = 30.0 * M;

// Initial positions (ensure non-planar motion)
const double x1_0 = (m2 / M) * rsc;                 // x-position of black hole 1
const double y1_0 = 0.0;                            // y-position of black hole 1
const double z1_0 = 0.0;                            // z-position of black hole 1
const double x2_0 = -(m1 / M) * rsc;                // x-position of black hole 2
const double y2_0 = 0.0;                            // y-position of black hole 2
const double z2_0 = 0.0;                            // z-position of black hole 2

// Relative separation
const double r_sep = sqrt((x1_0 - x2_0) * (x1_0 - x2_0) + (y1_0 - y2_0) * (y1_0 - y2_0) + (z1_0 - z2_0) * (z1_0 - z2_0));
const double v0 = sqrt(M / r_sep);

// Initial velocities
const double vx1_0 = 0.0;                           // x-velocity of black hole 1
const double vy1_0 = (m2 / M) * v0;                 // y-velocity of black hole 1
const double vz1_0 = 0.0;                           // z-velocity of black hole 1
const double vx2_0 = 0.0;                           // x-velocity of black hole 2   
const double vy2_0 = -(m1 / M) * v0;                // y-velocity of black hole 2
const double vz2_0 = 0.0;                           // z-velocity of black hole 2

// Spin parameters
const double chi1 = -1.0;                        // Spin parameter for black hole 1  
const double chi2 = 0.5;                        // Spin parameter for black hole 2

// Initial spin orientation vectors
const double vec_spin1[3] = {2.0/3.0, -1.0/3.0, -2.0/3.0};    // Spin orientation (vector components) for black hole 1
const double vec_spin2[3] = {-2.0/7.0, -3.0/7.0, 6.0/7.0};    // Spin orientation (vector components) for black hole 2

// Initial spin vectors (proper spin)
const double Sp1_0x = G * m1 * m1 * vec_spin1[0] * chi1 / c;     // x-component of spin for black hole 1
const double Sp1_0y = G * m1 * m1 * vec_spin1[1] * chi1 / c;     // y-component of spin for black hole 1
const double Sp1_0z = G * m1 * m1 * vec_spin1[2] * chi1 / c;     // z-component of spin for black hole 1
const double Sp2_0x = G * m2 * m2 * vec_spin2[0] * chi2 / c;     // x-component of spin for black hole 2
const double Sp2_0y = G * m2 * m2 * vec_spin2[1] * chi2 / c;     // y-component of spin for black hole 2
const double Sp2_0z = G * m2 * m2 * vec_spin2[2] * chi2 / c;     // z-component of spin for black hole 2



// Initial conditions
struct State {
    double x1, y1, z1, vx1, vy1, vz1;
    double x2, y2, z2, vx2, vy2, vz2;
    double Sp1x, Sp1y, Sp1z, Sp2x, Sp2y, Sp2z;
};

// Define various mathematical operations (cross product, dot product, norm)
std::array<double, 3> cross(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
double norm(const std::array<double, 3>& a) {
    return sqrt(dot(a, a));
}

State operator+(const State& a, const State& b) {
    State r = a;
    r.x1 += b.x1; r.y1 += b.y1; r.z1 += b.z1;
    r.vx1 += b.vx1; r.vy1 += b.vy1; r.vz1 += b.vz1;
    r.x2 += b.x2; r.y2 += b.y2; r.z2 += b.z2;
    r.vx2 += b.vx2; r.vy2 += b.vy2; r.vz2 += b.vz2;
    r.Sp1x += b.Sp1x; r.Sp1y += b.Sp1y; r.Sp1z += b.Sp1z;
    r.Sp2x += b.Sp2x; r.Sp2y += b.Sp2y; r.Sp2z += b.Sp2z;
    return r;
}
State operator-(const State& a, const State& b) {
    State r = a;
    r.x1 -= b.x1; r.y1 -= b.y1; r.z1 -= b.z1;
    r.vx1 -= b.vx1; r.vy1 -= b.vy1; r.vz1 -= b.vz1;
    r.x2 -= b.x2; r.y2 -= b.y2; r.z2 -= b.z2;
    r.vx2 -= b.vx2; r.vy2 -= b.vy2; r.vz2 -= b.vz2;
    r.Sp1x -= b.Sp1x; r.Sp1y -= b.Sp1y; r.Sp1z -= b.Sp1z;
    r.Sp2x -= b.Sp2x; r.Sp2y -= b.Sp2y; r.Sp2z -= b.Sp2z;
    return r;
}
State operator*(const State& a, double s) {
    State r = a;
    r.x1 *= s; r.y1 *= s; r.z1 *= s;
    r.vx1 *= s; r.vy1 *= s; r.vz1 *= s;
    r.x2 *= s; r.y2 *= s; r.z2 *= s;
    r.vx2 *= s; r.vy2 *= s; r.vz2 *= s;
    r.Sp1x *= s; r.Sp1y *= s; r.Sp1z *= s;
    r.Sp2x *= s; r.Sp2y *= s; r.Sp2z *= s;
    return r;
}
State operator*(double s, const State& a) { return a * s; }

// Compute the norm of the difference between two states (for error estimate)
double state_norm(const State& a) {
    return sqrt(
        a.x1*a.x1 + a.y1*a.y1 + a.z1*a.z1 +
        a.vx1*a.vx1 + a.vy1*a.vy1 + a.vz1*a.vz1 +
        a.x2*a.x2 + a.y2*a.y2 + a.z2*a.z2 +
        a.vx2*a.vx2 + a.vy2*a.vy2 + a.vz2*a.vz2 +
        a.Sp1x*a.Sp1x + a.Sp1y*a.Sp1y + a.Sp1z*a.Sp1z +
        a.Sp2x*a.Sp2x + a.Sp2y*a.Sp2y + a.Sp2z*a.Sp2z
    );
}


// Reduced angular momentum
std::array<double, 3> reduced_angular_momentum(const std::array<double, 3>& rvec, const std::array<double, 3>& vvec) {
    return cross(rvec, vvec);
}


// Total angular momentum
std::array<double, 3> total_angular_momentum(const std::array<double, 3>& L, const std::array<double, 3>& Sp1, const std::array<double, 3>& Sp2) {
    // mu * L + Sp1 + Sp2
    std::array<double, 3> result = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        result[i] = mu * L[i] + Sp1[i] + Sp2[i];
    }
    return result;
}


// Newtonian acceleration
std::array<double, 3> acceleration_N(const std::array<double, 3>& rvec) {
    double r = norm(rvec);
    std::array<double, 3> nvec = {rvec[0]/r, rvec[1]/r, rvec[2]/r};
    std::array<double, 3> a_N = {-M / (r*r) * nvec[0], -M / (r*r) * nvec[1], -M / (r*r) * nvec[2]};
    return a_N;
}

// 1PN acceleration
std::array<double, 3> acceleration_1PN(const std::array<double, 3>& rvec, const std::array<double, 3>& nvec, const std::array<double, 3>& vvec, double v, double vr) {
    double r = norm(rvec);
    double factor = -M / (r * r);

    // Compute scalar coefficients
    double coeff1 = (1.0 + 3.0 * nu) * v * v - 2.0 * (2.0 + nu) * M / r - 1.5 * nu * vr * vr;
    double coeff2 = -2.0 * (2.0 - nu) * vr;

    // nvec * coeff1
    std::array<double, 3> term1 = {nvec[0] * coeff1, nvec[1] * coeff1, nvec[2] * coeff1};
    // vvec * coeff2
    std::array<double, 3> term2 = {vvec[0] * coeff2, vvec[1] * coeff2, vvec[2] * coeff2};

    // a_1PN = factor * (term1 + term2)
    std::array<double, 3> a_1PN = {
        factor * (term1[0] + term2[0]),
        factor * (term1[1] + term2[1]),
        factor * (term1[2] + term2[2])
    };
    return a_1PN;
}

// Spin-orbit acceleration (1.5PN)
std::array<double, 3> acceleration_SO(
    const std::array<double, 3>& Sp,
    const std::array<double, 3>& L,
    double r,
    const std::array<double, 3>& nvec,
    double vr,
    const std::array<double, 3>& vvec,
    const std::array<double, 3>& xi
) {
    // Compute (4*Sp + 3*xi)
    std::array<double, 3> spin_combo = {
        4.0 * Sp[0] + 3.0 * xi[0],
        4.0 * Sp[1] + 3.0 * xi[1],
        4.0 * Sp[2] + 3.0 * xi[2]
    };

    // Dot product L · (4*Sp + 3*xi)
    double L_dot_spin = dot(L, spin_combo);

    // nvec / (2*r)
    std::array<double, 3> nvec_over_2r = {
        nvec[0] / (2.0 * r),
        nvec[1] / (2.0 * r),
        nvec[2] / (2.0 * r)
    };

    // 3 * nvec / (2*r) * (L · (4*Sp + 3*xi))
    std::array<double, 3> term1 = {
        3.0 * nvec_over_2r[0] * L_dot_spin,
        3.0 * nvec_over_2r[1] * L_dot_spin,
        3.0 * nvec_over_2r[2] * L_dot_spin
    };

    // - cross(vvec, (4*Sp + 3*xi))
    std::array<double, 3> term2 = cross(vvec, spin_combo);
    term2[0] *= -1.0; term2[1] *= -1.0; term2[2] *= -1.0;

    // 3 * vr * cross(nvec, (4*Sp + 3*xi)) / 2
    std::array<double, 3> cross_n_spin = cross(nvec, spin_combo);
    std::array<double, 3> term3 = {
        1.5 * vr * cross_n_spin[0],
        1.5 * vr * cross_n_spin[1],
        1.5 * vr * cross_n_spin[2]
    };

    // Sum all terms
    std::array<double, 3> sum = {
        term1[0] + term2[0] + term3[0],
        term1[1] + term2[1] + term3[1],
        term1[2] + term2[2] + term3[2]
    };

    // Multiply by 1/r^3
    double inv_r3 = 1.0 / (r * r * r);
    std::array<double, 3> a_SO = {
        inv_r3 * sum[0],
        inv_r3 * sum[1],
        inv_r3 * sum[2]
    };

    return a_SO;
}

// 2.5PN acceleration
std::array<double, 3> acceleration_2_5PN(
    double r,
    const std::array<double, 3>& nvec,
    double vr,
    double v,
    const std::array<double, 3>& vvec
) {
    double prefactor = 8.0 * nu * M * M / (5.0 * r * r * r);
    double coeff_n = vr * (3.0 * v * v + 17.0 * M / (3.0 * r));
    double coeff_v = v * v + 3.0 * M / r;

    std::array<double, 3> term_n = {nvec[0] * coeff_n, nvec[1] * coeff_n, nvec[2] * coeff_n};
    std::array<double, 3> term_v = {vvec[0] * coeff_v, vvec[1] * coeff_v, vvec[2] * coeff_v};

    std::array<double, 3> a_2_5PN = {
        prefactor * (term_n[0] - term_v[0]),
        prefactor * (term_n[1] - term_v[1]),
        prefactor * (term_n[2] - term_v[2])
    };
    return a_2_5PN;
}


// 3.5PN acceleration
std::array<double, 3> acceleration_3_5PN(
    double vr,
    double r,
    double v,
    const std::array<double, 3>& L,
    const std::array<double, 3>& nvec,
    const std::array<double, 3>& vvec,
    const std::array<double, 3>& Sp,
    const std::array<double, 3>& xi
) {
    double r2 = r * r;
    double r3 = r2 * r;
    double r4 = r3 * r;
    double v2 = v * v;
    double v4 = v2 * v2;
    double vr2 = vr * vr;
    double vr4 = vr2 * vr2;

    double LdotSp = dot(L, Sp);
    double LdotXi = dot(L, xi);

    // vr * nvec / r * ( ... )
    double coeff_n = (120.0 * v2 + 280.0 * vr2 + 453.0 * M / r) * LdotSp
                   + (285.0 * v2 - 245.0 * vr2 + 211.0 * M / r) * LdotXi;
    std::array<double, 3> term_n = {
        vr * nvec[0] / r * coeff_n,
        vr * nvec[1] / r * coeff_n,
        vr * nvec[2] / r * coeff_n
    };

    // vvec / r * ( ... )
    double coeff_v = (87.0 * v2 - 675.0 * vr2 - 901.0 * M / (3.0 * r)) * LdotSp
                   + 4.0 * (6.0 * v2 - 75.0 * vr2 - 41.0 * M / r) * LdotXi;
    std::array<double, 3> term_v = {
        vvec[0] / r * coeff_v,
        vvec[1] / r * coeff_v,
        vvec[2] / r * coeff_v
    };

    // -2 * vr * cross(vvec, Sp) / 3 * (48 * v^2 + 15 * vr^2 + 364 * M / r)
    double coeff_cross_sp = (48.0 * v2 + 15.0 * vr2 + 364.0 * M / r);
    std::array<double, 3> cross_v_sp = cross(vvec, Sp);
    std::array<double, 3> term_cross_sp = {
        -2.0 * vr / 3.0 * cross_v_sp[0] * coeff_cross_sp,
        -2.0 * vr / 3.0 * cross_v_sp[1] * coeff_cross_sp,
        -2.0 * vr / 3.0 * cross_v_sp[2] * coeff_cross_sp
    };

    // -vr * cross(vvec, xi) / 3 * (375 * v^2 - 195 * vr^2 + 640 * M / r)
    double coeff_cross_xi = (375.0 * v2 - 195.0 * vr2 + 640.0 * M / r);
    std::array<double, 3> cross_v_xi = cross(vvec, xi);
    std::array<double, 3> term_cross_xi = {
        -vr / 3.0 * cross_v_xi[0] * coeff_cross_xi,
        -vr / 3.0 * cross_v_xi[1] * coeff_cross_xi,
        -vr / 3.0 * cross_v_xi[2] * coeff_cross_xi
    };

    // cross(nvec, Sp) / 2 * (31 * v^4 - 260 * v^2 * vr^2 + 245 * vr^4 - 689 * v^2 * M / (3 * r) + 537 * vr^2 * M / r + 4 * M^2 / (3 * r^2))
    double coeff_n_sp = (31.0 * v4 - 260.0 * v2 * vr2 + 245.0 * vr4
                        - 689.0 * v2 * M / (3.0 * r) + 537.0 * vr2 * M / r + 4.0 * M * M / (3.0 * r2));
    std::array<double, 3> cross_n_sp = cross(nvec, Sp);
    std::array<double, 3> term_n_sp = {
        0.5 * cross_n_sp[0] * coeff_n_sp,
        0.5 * cross_n_sp[1] * coeff_n_sp,
        0.5 * cross_n_sp[2] * coeff_n_sp
    };

    // -cross(nvec, xi) / 2 * (29 * v^4 - 40 * v^2 * vr^2 - 245 * vr^4 + 211 * v^2 * M / r - 1019 * vr^2 * M / r - 80 * M^2 / r^2)
    double coeff_n_xi = (29.0 * v4 - 40.0 * v2 * vr2 - 245.0 * vr4
                        + 211.0 * v2 * M / r - 1019.0 * vr2 * M / r - 80.0 * M * M / r2);
    std::array<double, 3> cross_n_xi = cross(nvec, xi);
    std::array<double, 3> term_n_xi = {
        -0.5 * cross_n_xi[0] * coeff_n_xi,
        -0.5 * cross_n_xi[1] * coeff_n_xi,
        -0.5 * cross_n_xi[2] * coeff_n_xi
    };

    // Sum all terms
    std::array<double, 3> sum = {
        term_n[0] + term_v[0] + term_cross_sp[0] + term_cross_xi[0] + term_n_sp[0] + term_n_xi[0],
        term_n[1] + term_v[1] + term_cross_sp[1] + term_cross_xi[1] + term_n_sp[1] + term_n_xi[1],
        term_n[2] + term_v[2] + term_cross_sp[2] + term_cross_xi[2] + term_n_sp[2] + term_n_xi[2]
    };

    double prefactor = -nu * M / (5.0 * r4);

    std::array<double, 3> a_3_5PN = {
        prefactor * sum[0],
        prefactor * sum[1],
        prefactor * sum[2]
    };

    return a_3_5PN;
}

// Kerr ISCO radius for Kerr black holes
// sign = +1 for retrograde (L·S < 0), -1 for prograde (L·S > 0)
double KerrISCO(double M, double chi_eff_kerr, double Z1, double Z2, int sign=1) {
    return M * (3.0 + Z2 + sign * sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}


// Equations of motion for ODE integration
void derivatives(const State& y, State& dydt) {
    // Extract positions, velocities, and spins
    std::array<double, 3> rvec = {y.x1 - y.x2, y.y1 - y.y2, y.z1 - y.z2};
    std::array<double, 3> vvec = {y.vx1 - y.vx2, y.vy1 - y.vy2, y.vz1 - y.vz2};
    double r = norm(rvec);
    std::array<double, 3> nvec = {rvec[0]/r, rvec[1]/r, rvec[2]/r};
    double v = norm(vvec);
    double vr = dot(nvec, vvec);

    std::array<double, 3> Sp1 = {y.Sp1x, y.Sp1y, y.Sp1z};
    std::array<double, 3> Sp2 = {y.Sp2x, y.Sp2y, y.Sp2z};
    std::array<double, 3> Sp = {Sp1[0] + Sp2[0], Sp1[1] + Sp2[1], Sp1[2] + Sp2[2]};
    std::array<double, 3> L = reduced_angular_momentum(rvec, vvec);
    std::array<double, 3> xi = {
        (m2 * Sp1[0] / m1) + (m1 * Sp2[0] / m2),
        (m2 * Sp1[1] / m1) + (m1 * Sp2[1] / m2),
        (m2 * Sp1[2] / m1) + (m1 * Sp2[2] / m2)
    };

    // Accelerations
    std::array<double, 3> a_N = acceleration_N(rvec);
    std::array<double, 3> a_1PN = acceleration_1PN(rvec, nvec, vvec, v, vr);
    std::array<double, 3> a_SO = acceleration_SO(Sp, L, r, nvec, vr, vvec, xi);
    std::array<double, 3> a_2_5PN = acceleration_2_5PN(r, nvec, vr, v, vvec);
    std::array<double, 3> a_3_5PN = acceleration_3_5PN(vr, r, v, L, nvec, vvec, Sp, xi);

    // Total acceleration
    std::array<double, 3> a_total = {
        a_N[0] + a_1PN[0] + a_SO[0] + a_2_5PN[0] + a_3_5PN[0],
        a_N[1] + a_1PN[1] + a_SO[1] + a_2_5PN[1] + a_3_5PN[1],
        a_N[2] + a_1PN[2] + a_SO[2] + a_2_5PN[2] + a_3_5PN[2]
    };

    // Accelerations for each BH
    dydt.vx1 = m2 / M * a_total[0];
    dydt.vy1 = m2 / M * a_total[1];
    dydt.vz1 = m2 / M * a_total[2];

    dydt.vx2 = -m1 / M * a_total[0];
    dydt.vy2 = -m1 / M * a_total[1];
    dydt.vz2 = -m1 / M * a_total[2];

    // Positions
    dydt.x1 = y.vx1;
    dydt.y1 = y.vy1;
    dydt.z1 = y.vz1;
    dydt.x2 = y.vx2;
    dydt.y2 = y.vy2;
    dydt.z2 = y.vz2;

    // Spin evolution (precession equations)
    double spin_fac1 = nu * M / (r*r*r) * (2.0 + 1.5 * m2 / m1);
    double spin_fac2 = nu * M / (r*r*r) * (2.0 + 1.5 * m1 / m2);
    std::array<double, 3> dSp1dt = cross(L, Sp1);
    std::array<double, 3> dSp2dt = cross(L, Sp2);

    dydt.Sp1x = spin_fac1 * dSp1dt[0];
    dydt.Sp1y = spin_fac1 * dSp1dt[1];
    dydt.Sp1z = spin_fac1 * dSp1dt[2];
    dydt.Sp2x = spin_fac2 * dSp2dt[0];
    dydt.Sp2y = spin_fac2 * dSp2dt[1];
    dydt.Sp2z = spin_fac2 * dSp2dt[2];
}


void rk4_step(State& y, double dt) {
    State k1, k2, k3, k4, y_temp;

    derivatives(y, k1);

    y_temp = y;
    y_temp.x1 += 0.5 * dt * k1.x1; y_temp.y1 += 0.5 * dt * k1.y1; y_temp.z1 += 0.5 * dt * k1.z1;
    y_temp.vx1 += 0.5 * dt * k1.vx1; y_temp.vy1 += 0.5 * dt * k1.vy1; y_temp.vz1 += 0.5 * dt * k1.vz1;
    y_temp.x2 += 0.5 * dt * k1.x2; y_temp.y2 += 0.5 * dt * k1.y2; y_temp.z2 += 0.5 * dt * k1.z2;
    y_temp.vx2 += 0.5 * dt * k1.vx2; y_temp.vy2 += 0.5 * dt * k1.vy2; y_temp.vz2 += 0.5 * dt * k1.vz2;
    y_temp.Sp1x += 0.5 * dt * k1.Sp1x; y_temp.Sp1y += 0.5 * dt * k1.Sp1y; y_temp.Sp1z += 0.5 * dt * k1.Sp1z;
    y_temp.Sp2x += 0.5 * dt * k1.Sp2x; y_temp.Sp2y += 0.5 * dt * k1.Sp2y; y_temp.Sp2z += 0.5 * dt * k1.Sp2z;
    derivatives(y_temp, k2);

    y_temp = y;
    y_temp.x1 += 0.5 * dt * k2.x1; y_temp.y1 += 0.5 * dt * k2.y1; y_temp.z1 += 0.5 * dt * k2.z1;
    y_temp.vx1 += 0.5 * dt * k2.vx1; y_temp.vy1 += 0.5 * dt * k2.vy1; y_temp.vz1 += 0.5 * dt * k2.vz1;
    y_temp.x2 += 0.5 * dt * k2.x2; y_temp.y2 += 0.5 * dt * k2.y2; y_temp.z2 += 0.5 * dt * k2.z2;
    y_temp.vx2 += 0.5 * dt * k2.vx2; y_temp.vy2 += 0.5 * dt * k2.vy2; y_temp.vz2 += 0.5 * dt * k2.vz2;
    y_temp.Sp1x += 0.5 * dt * k2.Sp1x; y_temp.Sp1y += 0.5 * dt * k2.Sp1y; y_temp.Sp1z += 0.5 * dt * k2.Sp1z;
    y_temp.Sp2x += 0.5 * dt * k2.Sp2x; y_temp.Sp2y += 0.5 * dt * k2.Sp2y; y_temp.Sp2z += 0.5 * dt * k2.Sp2z;
    derivatives(y_temp, k3);

    y_temp = y;
    y_temp.x1 += dt * k3.x1; y_temp.y1 += dt * k3.y1; y_temp.z1 += dt * k3.z1;
    y_temp.vx1 += dt * k3.vx1; y_temp.vy1 += dt * k3.vy1; y_temp.vz1 += dt * k3.vz1;
    y_temp.x2 += dt * k3.x2; y_temp.y2 += dt * k3.y2; y_temp.z2 += dt * k3.z2;
    y_temp.vx2 += dt * k3.vx2; y_temp.vy2 += dt * k3.vy2; y_temp.vz2 += dt * k3.vz2;
    y_temp.Sp1x += dt * k3.Sp1x; y_temp.Sp1y += dt * k3.Sp1y; y_temp.Sp1z += dt * k3.Sp1z;
    y_temp.Sp2x += dt * k3.Sp2x; y_temp.Sp2y += dt * k3.Sp2y; y_temp.Sp2z += dt * k3.Sp2z;
    derivatives(y_temp, k4);

    // Update y with weighted sum
    y.x1 += dt / 6.0 * (k1.x1 + 2*k2.x1 + 2*k3.x1 + k4.x1);
    y.y1 += dt / 6.0 * (k1.y1 + 2*k2.y1 + 2*k3.y1 + k4.y1);
    y.z1 += dt / 6.0 * (k1.z1 + 2*k2.z1 + 2*k3.z1 + k4.z1);
    y.vx1 += dt / 6.0 * (k1.vx1 + 2*k2.vx1 + 2*k3.vx1 + k4.vx1);
    y.vy1 += dt / 6.0 * (k1.vy1 + 2*k2.vy1 + 2*k3.vy1 + k4.vy1);
    y.vz1 += dt / 6.0 * (k1.vz1 + 2*k2.vz1 + 2*k3.vz1 + k4.vz1);
    y.x2 += dt / 6.0 * (k1.x2 + 2*k2.x2 + 2*k3.x2 + k4.x2);
    y.y2 += dt / 6.0 * (k1.y2 + 2*k2.y2 + 2*k3.y2 + k4.y2);
    y.z2 += dt / 6.0 * (k1.z2 + 2*k2.z2 + 2*k3.z2 + k4.z2);
    y.vx2 += dt / 6.0 * (k1.vx2 + 2*k2.vx2 + 2*k3.vx2 + k4.vx2);
    y.vy2 += dt / 6.0 * (k1.vy2 + 2*k2.vy2 + 2*k3.vy2 + k4.vy2);
    y.vz2 += dt / 6.0 * (k1.vz2 + 2*k2.vz2 + 2*k3.vz2 + k4.vz2);
    y.Sp1x += dt / 6.0 * (k1.Sp1x + 2*k2.Sp1x + 2*k3.Sp1x + k4.Sp1x);
    y.Sp1y += dt / 6.0 * (k1.Sp1y + 2*k2.Sp1y + 2*k3.Sp1y + k4.Sp1y);
    y.Sp1z += dt / 6.0 * (k1.Sp1z + 2*k2.Sp1z + 2*k3.Sp1z + k4.Sp1z);
    y.Sp2x += dt / 6.0 * (k1.Sp2x + 2*k2.Sp2x + 2*k3.Sp2x + k4.Sp2x);
    y.Sp2y += dt / 6.0 * (k1.Sp2y + 2*k2.Sp2y + 2*k3.Sp2y + k4.Sp2y);
    y.Sp2z += dt / 6.0 * (k1.Sp2z + 2*k2.Sp2z + 2*k3.Sp2z + k4.Sp2z);
}

// Adaptive RK4 step with step-doubling
bool rk4_adaptive_step(State& y, double& t, double& dt, double tol) {
    // Take one full step of size dt
    State y_full = y;
    State y_temp = y;
    rk4_step(y_full, dt);

    // Take two half steps
    double half_dt = 0.5 * dt;
    rk4_step(y_temp, half_dt);
    rk4_step(y_temp, half_dt);

    // Estimate error
    State diff = y_full - y_temp;
    double err = state_norm(diff);

    // Accept step if error is small enough
    if (err < tol) {
        y = y_temp;
        t += dt;
        // Increase step size for next step
        dt *= std::min(2.0, 0.9 * pow(tol / (err + 1e-16), 0.2));
        return true;
    } else {
        // Reject step, decrease dt and try again
        dt *= std::max(0.1, 0.9 * pow(tol / (err + 1e-16), 0.25));
        return false;
    }
}

// RK_4 adaptive integration with merger condition
void integrate_adaptive(State& y, double t0, double t1, double dt_init, double tol, ofstream& out) {
    double t = t0;
    double dt = dt_init;
    int max_steps = 10000000;
    int steps = 0;
    while (t < t1 && steps < max_steps) {
        // Check merger condition (as before)
        std::array<double, 3> rvec = {y.x1 - y.x2, y.y1 - y.y2, y.z1 - y.z2};
        std::array<double, 3> vvec = {y.vx1 - y.vx2, y.vy1 - y.vy2, y.vz1 - y.vz2};
        double dist = norm(rvec);

        std::array<double, 3> Sp1 = {y.Sp1x, y.Sp1y, y.Sp1z};
        std::array<double, 3> Sp2 = {y.Sp2x, y.Sp2y, y.Sp2z};
        std::array<double, 3> Sp = {Sp1[0] + Sp2[0], Sp1[1] + Sp2[1], Sp1[2] + Sp2[2]};
        std::array<double, 3> L = reduced_angular_momentum(rvec, vvec);

        double L_norm = norm(L) + 1e-16;
        std::array<double, 3> L_hat = {L[0]/L_norm, L[1]/L_norm, L[2]/L_norm};
        double S_tot_dot_Lhat = dot(Sp, L_hat);
        double chi_eff_kerr = S_tot_dot_Lhat / (M * M);
        if (chi_eff_kerr > 1.0) chi_eff_kerr = 1.0;
        if (chi_eff_kerr < -1.0) chi_eff_kerr = -1.0;
        double LS_dot = dot(L, Sp);
        int sign = (LS_dot > 0) ? -1 : +1;
        double Z1 = 1.0 + pow(1.0 - chi_eff_kerr * chi_eff_kerr, 1.0/3.0) *
                    (pow(1.0 + chi_eff_kerr, 1.0/3.0) + pow(1.0 - chi_eff_kerr, 1.0/3.0));
        double Z2 = sqrt(3.0 * chi_eff_kerr * chi_eff_kerr + Z1 * Z1);
        double r_ISCO = KerrISCO(M, chi_eff_kerr, Z1, Z2, sign);

        if (dist < 2.0 * r_ISCO) {
            cout << "Merger detected at t = " << t << endl;
            cout << "ISCO radius (M): " << r_ISCO << ", current separation: " << dist << ", dist/2: " << dist/2 << endl;
            break;
        }

        // Adaptive RK4 step
        bool step_accepted = rk4_adaptive_step(y, t, dt, tol);
        if (step_accepted) {
            // Output for animation
            out << y.x1 << "," << y.y1 << "," << y.z1 << ","
                << y.x2 << "," << y.y2 << "," << y.z2 << ","
                << y.vx1 << "," << y.vy1 << "," << y.vz1 << ","
                << y.vx2 << "," << y.vy2 << "," << y.vz2 << ","
                << y.Sp1x << "," << y.Sp1y << "," << y.Sp1z << ","
                << y.Sp2x << "," << y.Sp2y << "," << y.Sp2z << ","
                << m1 << "," << m2 << "\n";
            steps++;
        }
        // If not accepted, dt is reduced and step is retried
    }
}



int main() {
    State y;
    // Initial conditions
    y.x1 = x1_0; y.y1 = y1_0; y.z1 = z1_0;
    y.vx1 = vx1_0; y.vy1 = vy1_0; y.vz1 = vz1_0;
    y.x2 = x2_0; y.y2 = y2_0; y.z2 = z2_0;
    y.vx2 = vx2_0; y.vy2 = vy2_0; y.vz2 = vz2_0;
    y.Sp1x = Sp1_0x; y.Sp1y = Sp1_0y; y.Sp1z = Sp1_0z;
    y.Sp2x = Sp2_0x; y.Sp2y = Sp2_0y; y.Sp2z = Sp2_0z;

    ofstream out("orbit.csv");
    integrate_adaptive(y, 0.0, 1000000.0, 1.0, 1e-14, out);
    out.close();
    cout << "Simulation complete. Data written to orbit.csv\n";
    return 0;
}
