# PJL Thesis Scripts Data

A collection of quantum physics simulation scripts for modeling Barium atom dynamics using optical transitions and laser interactions. The simulations are implemented in both MATLAB and Python.

## Overview

This repository contains computational models for simulating atomic energy level dynamics, optical pumping, fluorescence, and Rabi oscillations in Barium atoms. The work is focused on optical control and characterization of atomic systems, particularly relevant for quantum information processing and atomic spectroscopy.

## Project Structure

### `Simulation_Scripts/`

Code is organized by programming language with parallel implementations in MATLAB and Python.

```
Simulation_Scripts/
├── MATLAB/
│   ├── Ba_4Levels_493_650_1762_Lasers_Sim/
│   ├── Ba_Fluorescence_Simulations/
│   ├── Ba_OpticalPump_ToyModel_Sim/
│   └── Thermal_Ba_RabiFlop/
└── Python/
    ├── Ba_4Levels_493_650_1762_Lasers_Sim/
    ├── Ba_Fluorescence_Simulations/
    ├── Ba_OpticalPump_ToyModel_Sim/
    └── Thermal_Ba_RabiFlop/
```

Each language folder contains four main simulation categories:

#### 1. **Ba_4Levels_493_650_1762_Lasers_Sim/**

Simulations of a simplified 4-level Barium energy level system involving the S₁/₂, P₁/₂, D₃/₂, and D₅/₂ states using three laser wavelengths (493 nm, 650 nm, and 1762 nm).

**Key Scripts:**

MATLAB (`MATLAB/Ba_4Levels_493_650_1762_Lasers_Sim/`):
- `Ba_4Levels_Shelving_Sim.m`: Time-resolved evolution of state populations using Runge-Kutta integration
- `Ba_4Levels_Shelving_Sim_linsolve.m`: Steady-state solution from linear system
- `Misc_Ba_4Levels_FreqScan.m`: Frequency scan analysis of 1762 nm laser

Python (`Python/Ba_4Levels_493_650_1762_Lasers_Sim/`):
- `ba_4levels_shelving_sim.py`: Time-resolved evolution of state populations using Runge-Kutta integration
- `ba_4levels_shelving_sim_linsolve.py`: Steady-state solution from linear system
- `misc_ba_4levels_freq_scan.py`: Frequency scan analysis of 1762 nm laser
- `sim.ipynb`: Jupyter notebook with visualizations and analysis

**Applications:**
- Frequency scanning of laser detuning
- Population dynamics in multi-level shelving systems
- Optical pumping efficiency analysis

---

#### 2. **Ba_Fluorescence_Simulations/**

Comprehensive models of spontaneous emission and fluorescence in Barium S-P-D level systems, including Zeeman structure for Ba-137 and Ba-138 isotopes.

**Core Models:**

**Simple S-P-D System:**

MATLAB (`MATLAB/Ba_Fluorescence_Simulations/`):
- `Ba_SPD_Levels_OpticalBloch_Analytical.m`: Analytical solution for P-state population
- `Ba_SPD_Levels_OpticalBloch_Analytical_DeltaScan.m`: Frequency detuning scans
- `Ba_SPD_Levels_OpticalBloch_linsolve.m`: Linear system solver

Python (`Python/Ba_Fluorescence_Simulations/`):
- `ba_spd_levels_optical_bloch_analytical.py`: Analytical solution for P-state population
- `ba_spd_levels_optical_bloch_analytical_delta_scan.py`: Frequency detuning scans
- `ba_spd_levels_optical_bloch_linsolve.py`: Linear system solver

**Ba-137 Zeeman Structure:**

MATLAB:
- `Ba137_SPDZeeman_Levels_OpticalBloch_linsolve_prototype.m`: Solves steady-state density matrix equations
- `Ba137_SPDZeeman_Levels_OpticalBloch_init_prototype.m`: Initializes Clebsch-Gordan coefficients and branching ratios
- `Ba137_SPDZeeman_Levels_OpticalBloch_DeltaScan.m`: Laser frequency detuning scans
- `Ba137_SPDZeeman_Levels_OpticalBloch_Sim_LambdaDetuneScan.m`: Sideband frequency relative detuning scans
- `Ba137_SPDZeeman_Levels_OpticalBloch_Sim_PowerScan.m`: Rabi frequency (laser power) variation analysis
- `Ba137_SPDZeeman_Levels_OpticalBloch_Sim_PolarizationScan.m`: Laser polarization parameter studies

Python:
- `ba137_spd_zeeman_levels_optical_bloch_linsolve_prototype.py`: Solves steady-state density matrix equations
- `ba137_spd_zeeman_levels_optical_bloch_init_prototype.py`: Initializes Clebsch-Gordan coefficients and branching ratios
- `ba137_spd_zeeman_levels_optical_bloch_delta_scan.py`: Laser frequency detuning scans
- `ba137_spd_zeeman_levels_optical_bloch_sim_lambda_detune_scan.py`: Sideband frequency relative detuning scans
- `ba137_spd_zeeman_levels_optical_bloch_sim_power_scan.py`: Rabi frequency (laser power) variation analysis
- `ba137_spd_zeeman_levels_optical_bloch_sim_polarization_scan.py`: Laser polarization parameter studies

**Ba-138 Zeeman Structure:**

MATLAB:
- `Ba138_SPDZeeman_Levels_OpticalBloch_linsolve.m`: Solves steady-state density matrix equations
- `Ba138_SPDZeeman_Levels_OpticalBloch_linsolve_v2.m`: Alternative version with shifted energy offsets for numerical stability
- `Ba138_SPDZeeman_Levels_OpticalBloch_DeltaScan.m`: Laser frequency detuning scans
- `Ba138_SPDZeeman_Levels_OpticalBloch_Sim_PowerScan_v2.m`: Rabi frequency variation analysis
- `Ba138_SPDZeeman_Levels_OpticalBloch_Sim_PolarizationScan_v2.m`: Laser polarization studies

Python:
- `ba138_spd_zeeman_levels_optical_bloch_linsolve.py`: Solves steady-state density matrix equations
- `ba138_spd_zeeman_levels_optical_bloch_linsolve_v2.py`: Alternative version with shifted energy offsets
- `ba138_spd_zeeman_levels_optical_bloch_delta_scan.py`: Laser frequency detuning scans
- `ba138_spd_zeeman_levels_optical_bloch_sim_power_scan_v2.py`: Rabi frequency variation analysis
- `ba138_spd_zeeman_levels_optical_bloch_sim_polarization_scan_v2.py`: Laser polarization studies

**Utility Functions:**

MATLAB & Python (both languages):
- `FmtoJCoefficient.m` / `fm_to_j_coefficient.py`: Wigner-Eckart coefficients for |F,m⟩ → |J⟩ space reduction
- `GetClebschGordan.m` / `get_clebsch_gordan.py`: Clebsch-Gordan coefficient library generation
- `NotWigner6j.m` / `not_wigner_6j.py`: Wigner-6j related coefficients for matrix element reduction

---

#### 3. **Ba_OpticalPump_ToyModel_Sim/**

Simplified 4-level optical pumping model with 2 ground state and 2 excited state levels.

**Scripts:**

MATLAB (`MATLAB/Ba_OpticalPump_ToyModel_Sim/`):
- `Ba_OpticalPump_Sim_4Levels.m`: Steady-state population calculation

Python (`Python/Ba_OpticalPump_ToyModel_Sim/`):
- `ba_optical_pump_sim_4levels.py`: Steady-state population calculation

**Applications:**
- Fundamental optical pumping efficiency studies
- Basic dynamic decoupling sequences
- Simplified experimental design exploration

---

#### 4. **Thermal_Ba_RabiFlop/**

Models of Rabi oscillations in thermal Barium atom ensembles.

**Scripts:**

MATLAB (`MATLAB/Thermal_Ba_RabiFlop/`):
- `ThermalRSBRabiFlop.m`: Time-domain Rabi flopping dynamics
- `Getfunpara.m`: Helper functions for parameter management

Python (`Python/Thermal_Ba_RabiFlop/`):
- `thermal_rsb_rabi_flop.py`: Time-domain Rabi flopping dynamics
- `getfunpara.py`: Helper functions for parameter management

**Applications:**
- Rabi oscillation frequency measurements
- Red-sideband (RSB) cooling dynamics
- Thermal motion effects on coherent control

---

## Getting Started

### Prerequisites

**Python:**
```bash
pip install numpy scipy matplotlib
```

**MATLAB:**
- MATLAB 2016b or later recommended
- No additional toolboxes required for most scripts

### Running Simulations

**Python Example:**
```python
from Python.Ba_Fluorescence_Simulations.ba_spd_levels_optical_bloch_analytical import ba_spd_levels_optical_bloch_analytical
import numpy as np

# Define parameters
Delta_S = -2 * np.pi * 10  # S-P detuning (rad·MHz)
Delta_D = -2 * np.pi * 40  # D-P detuning (rad·MHz)
Omega_S = 2 * np.pi * 10   # S-P Rabi frequency (rad·MHz)
Omega_D = 2 * np.pi * 10   # D-P Rabi frequency (rad·MHz)

# Run simulation
sigma_PP = ba_spd_levels_optical_bloch_analytical(Delta_S, Delta_D, Omega_S, Omega_D)
print(f"P state population: {sigma_PP.real:.4f}")
```

**MATLAB Example:**
```matlab
% Example: Run Ba-138 linsolve simulation
cd Simulation_Scripts/MATLAB/Ba_Fluorescence_Simulations/
run Ba138_SPDZeeman_Levels_OpticalBloch_linsolve.m
```

**Jupyter Notebook:**
```bash
cd Simulation_Scripts/Python/Ba_4Levels_493_650_1762_Lasers_Sim/
jupyter notebook sim.ipynb
```

## Physics Background

### Key Concepts

**Optical Bloch Equations:** Master equations describing the evolution of density matrix elements under laser-atom interactions and spontaneous emission.

**Rabi Frequency (Ω):** Oscillation frequency of atomic state populations driven by a resonant laser field. Determined by laser intensity and atomic transition dipole moment.

**Detuning (Δ):** Frequency difference between laser and atomic transition. Δ = ω_laser - ω_transition.

**Decay Rate (γ):** Spontaneous emission rate of excited states. Affects coherence and population dynamics.

**Density Matrix (σ):** Complete quantum mechanical description of system state. Diagonal elements are populations, off-diagonal elements are coherences.

**Zeeman Splitting:** Energy level splitting due to magnetic fields, characterized by quantum numbers F and m_F for hyperfine structure.

**Clebsch-Gordan Coefficients:** Angular momentum coupling coefficients required for transitions between coupled electronic-nuclear angular momentum states.

### Physical Systems

**S₁/₂ → P₁/₂ → D₃/₂ Transitions:** Typical ladder system in Barium with fast spontaneous emission from P state (~95 MHz decay rate).

**Ba-137 vs Ba-138:** Different nuclear spins (I=3/2 for Ba-137, I=0 for Ba-138) lead to different hyperfine structure and Clebsch-Gordan coupling patterns.

## Output Files and Formats

Most simulations generate:
- **Population dynamics:** Time arrays and density matrix element trajectories
- **Frequency scans:** 1D/2D parameter sweeps showing steady-state populations
- **Steady-state solutions:** Equilibrium population distributions and coherences
- **Branching ratios:** Fluorescence pathway probabilities

## File Naming Conventions

- `_linsolve`: Linear system solver approach (direct steady-state)
- `_analytical`: Closed-form analytical solution
- `_delta_scan`: Frequency detuning parameter sweep
- `_power_scan`: Rabi frequency variation
- `_polarization_scan`: Laser polarization parameter study
- `_lambda_detune_scan`: Sideband frequency relative detuning
- `_prototype`: Prototype/test version
- `_v2`: Revised version with improvements or alternative approach

## Contributing

For modifications or extensions:
1. Maintain parallel Python/MATLAB implementations where possible
2. Update corresponding readme files
3. Document any new parameters or physics models
4. Include docstrings explaining key assumptions

## License

This repository is part of PJL thesis research. Please contact the author for usage information.

## Author

Thesis research code
Department: [Add department]
Laboratory: Senkolab

## References

For detailed documentation of individual simulations, see the `readme.txt` file in each subdirectory.

### Related Physics

- Optical Bloch Equations and Rate Equations
- Atomic Spectroscopy and Linewidth Theory
- Resonance Fluorescence
- Optical Pumping Dynamics
- Rabi Oscillations in Thermal Ensembles
- Hyperfine Coupling and Zeeman Effect
