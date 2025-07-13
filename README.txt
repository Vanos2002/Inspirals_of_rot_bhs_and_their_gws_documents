Repository made public on July 13, 2025; last updated July 13, 2025.

This repository contains the code developed for the Bachelor’s thesis titled “Inspirals of Rotating Black Holes and Their Gravitational Waves” by Matěj Váňa, supervised by Mgr. Vojtěch Witzany, Dr. rer. nat., at the Institute of Theoretical Physics, Charles University.

The two Mathematica nootebooks (developed via Mathematica 14.0 for desktop) titled "eom_2PN_derivation_for_anim.nb" and "eom_2PN_derivation_plots.nb" were employed for the 2PN equations of motion derivation from the Hamiltonian provided in the paper: "Integrability of eccentric, spinning black hole binaries up to second post-Newtonian order" - Sashwat Tanay, Leo C. Stein, José T. Gálvez Ghersi - arXiv:2012.06586. 

- The notebook "eom_2PN_derivation_for_anim.nb" derives the respective equations of motion using 6 canonically conjugate coordinates (r, phi, theta, pr, pphi, ptheta) and spin vectors S1, S2. The corresponding Python implementation (developed in Python 3.12.6) can be found in the file "2PN_animation.py", containing solely the animation for the given initial parameters. Besides displaying the orbital motion, it also projects the spin vectors in their instantaneous direction.

- The notebook "eom_2PN_derivation_plots.nb" derives the equations of motion using only two conjugate coordinates (r,pr). Along the spin vectors S1 and S2, it defines the four remaining conjugate coordinates through the orbital angular momentum vector L. The corresponding numerical implementation can be found in the file "2PN_graphing.py". The code includes some graphs that were employed for the plotting in the thesis.

The latter four codes, titled "3_5PN_animation.py" and "inspirals_3_5PN_plots.py" and "3_5PN_rk4_adaptive.cpp", and "cpp_animation.py", correspond to the 3.5PN order implementation based on the work: "Post-Newtonian gravitational radiation and equations of motion via direct integration of the relaxed Einstein equations. III. Radiation reaction for binary systems with spinning bodies" - Clifford M. Will - arXiv:gr-qc/0502039. The equations of motion are directly presented in the article and therefore do not have to be derived.

- The code "3_5PN_animation.py" includes solely the animation for the corresponding initial parameters. Besides displaying the motion, it also projects the spin vectors in their instantaneous direction. Additionally, it prints the Kerr-ISCO distance in units of M and meters, and also prints the code time and real-time to coalescence.

- The code "inspirals_3_5PN_plots.py" includes some graphs used for plotting in the thesis. It also returns the complete unanimated trajectory.

- For longer simulations, we encourage employing the faster C++ version "3_5PN_rk4_adaptive.cpp", which uses the Runge-Kutta 4 adaptive method.

- The Python code "cpp_animation.py" is used to animate the physics defined in "3_5PN_rk4_adaptive.cpp" in the Matplotlib environment.

INFORMATION REGARDING RUNNING THE PYTHON CODES:

- Ensure that you have downloaded all necessary libraries that are imported at the beginning of each document!
- We recommend using a virtual environment in VSCode, and to obtain the uninstalled libraries, write the command line in the terminal: pip install <library-name>
- For the files "2PN_graphing.py" and "inspirals_3_5PN_plots.py", the runtime may be long, hence consider removing unnecessary plots
- For the C++ version, we recommend using a virtual environment with "3_5PN_rk4_adaptive.cpp" and "cpp_animation.py". Run the code by accessing the virtual environment and writing the following commands to the terminal:
1.
g++ 3_5PN_rk4_adaptive.cpp -o 3_5PN_rk4_adaptive
./3_5PN_rk4_adaptive
2.
python3 cpp_animation.py
