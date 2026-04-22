Ba_SPD_Levels_OpticalBloch_Analytical.m: a MATLAB script for analytically calculating the P state population of a S-P-D 3-level toy model, with the P state having a fast decay and S and D state being stable.

Ba_SPD_Levels_OpticalBloch_Analytical_DeltaScan.m: a MATLAB script for analytically calculating the P state populations of a S-P-D 3-level toy model as in Ba_SPD_Levels_OpticalBloch_Analytical.m for a range of simulated laser frequency detunings.

Ba_SPD_Levels_OpticalBloch_linsolve.m: a MATLAB script for solving for the P state population of a S-P-D 3-level toy model from a linear equation system, with the P state having a fast decay and S and D state being stable.

Ba137_SPDZeeman_Levels_OpticalBloch_DeltaScan.m: a MATLAB script for generating the steady-state density matrix elements of a Ba-137 ion for a range of laser frequency detunings.

Ba137_SPDZeeman_Levels_OpticalBloch_init_prototype.m: a MATLAB script for initializing the Clebsch-Gordan coefficients and branching ratios for all the transitions and decay paths in a Ba-137 ion. Has to be run before Ba137_SPDZeeman_Levels_OpticalBloch_linsolve_prototype.m.

Ba137_SPDZeeman_Levels_OpticalBloch_linsolve_prototype.m: a MATLAB script for solving for the steady-state density matrix elements of a Ba-137 ion.

Ba137_SPDZeeman_Levels_OpticalBloch_Sim_LambdaDetuneScan.m: a MATLAB script for generating the maximum P state populations of Ba-137 (at optimal carrier laser frequencies) for a range of relative detunings of the sideband frequencies within the S-P and D-P lasers.

Ba137_SPDZeeman_Levels_OpticalBloch_Sim_PolarizationScan.m: a MATLAB script for generating the maximum P state populations of Ba-137 (at optimal carrier laser frequencies) for a range of laser polarizations.

Ba137_SPDZeeman_Levels_OpticalBloch_Sim_PowerScan.m: a MATLAB script for generating the maximum P state populations of Ba-137 (at optimal carrier laser frequencies) for a range of laser Rabi frequencies.

Ba138_SPDZeeman_Levels_OpticalBloch_DeltaScan.m: a MATLAB script for generating the steady-state density matrix elements of a Ba-138 ion for a range of laser frequency detunings.

Ba138_SPDZeeman_Levels_OpticalBloch_linsolve.m: a MATLAB script for solving for the steady-state density matrix elements of a Ba-138 ion.

Ba138_SPDZeeman_Levels_OpticalBloch_linsolve_v2.m: a MATLAB script for solving for the steady-state density matrix elements of a Ba-138 ion, with a shifted energy offset in the Hamiltonian, which does not affect the physical dynamics, only for simplifying the mathematical expressions.

Ba138_SPDZeeman_Levels_OpticalBloch_Sim_PolarizationScan_v2.m: a MATLAB script for generating the maximum P state populations of Ba-138 (at optimal carrier laser frequencies) for a range of laser polarizations.

Ba138_SPDZeeman_Levels_OpticalBloch_Sim_PowerScan_v2.m: a MATLAB script for generating the maximum P state populations of Ba-138 (at optimal carrier laser frequencies) for a range of laser Rabi frequencies.

FmtoJCoefficient.m: a MATLAB function for computing the Wigner-Eckart coefficient for reducing a transition matrix element in the |F,m> space to the |J> space.

GetClebschGordan.m: a MATLAB function for generating a library of Clebsch-Gordan coefficients.

NotWigner6j.m: a MATLAB function for generating a coefficient for Wigner-Eckart transition matrix reduction, related to the Wigner-6j coefficient.