# Project: "Optical attosecond spectroscopy of disordered media"

- Official starting date of the project: 2025-04-01.
- Deadline for submitting master's thesis: 2026-03-01.

### Main expected outcomes
A systematic theoretical investigation of how disorder affects femtosecond-scale light-matter interactions in solids, focusing primarily on:

- **Light-driven motion of photoinjected carriers** — momentum relaxation and transport dynamics
- **Free-induction decay** — dephasing of coherent electron-hole oscillations (status: under investigation, may or may not yield conclusive results)
- **Photoinjection dynamics** — nonlinear carrier generation mechanisms (initial findings suggest minimal qualitative disorder signatures in time domain; energy-domain analysis may reveal additional insights)

The investigation employs **time-dependent density functional theory (TDDFT)** with an **orthogonal pump-probe protocol** as the key methodological innovation.

## Research Project Description

### 1. Overview and Context

This project investigates how atomic disorder influences femtosecond-scale light-matter interactions in solids using *ab initio* time-dependent density functional theory (TDDFT). The work addresses a critical gap: while disorder is ubiquitous in real materials (from thermal phonons to fully amorphous structures), its role in ultrafast nonlinear optical processes remains poorly understood. This understanding is essential for interpreting optical-field-resolved measurements with attosecond-scale temporal resolution, a frontier experimental technique at the Laboratory for Attosecond Physics.

The project focuses on quenched structural disorder—atomic positions that are static on the femtosecond timescale of observation. Materials studied range from crystalline silicon with small atomic displacements to fully amorphous silicon, enabling systematic investigation of how disorder strength affects ultrafast carrier dynamics.

### 2. Central Challenge: Isolating Disorder Effects on Carrier Transport

A fundamental obstacle in studying disorder with TDDFT emerged early in the project: **in supercell simulations with periodic boundary conditions, Bloch's theorem can produce non-decaying currents even in disordered structures**, because the simulation describes a superposition of well-defined Bloch states with conserved crystal momenta. This "drift current dilemma" obscures the physical momentum relaxation that disorder should induce.

The project addresses numerically simulating pump-probe measurements with orthogonally polarized pulses: An intense "pump" pulse creates a population of electrons and holes via nonlinear photoinjection. A simultaneous or delayed weak "probe" pulse, **polarized orthogonally** to the pump, accelerates these carriers. In isotropic or rotationally symmetric media (amorphous silicon, or cubic crystals averaged over orientations), the pump induces no current in the orthogonal direction. Therefore, **any current measured in the probe direction arises exclusively from the motion of photoinjected carriers driven by the probe field**. This cleanly separates carrier generation from transport, enabling direct observation of momentum relaxation dynamics and their dependence on disorder.

### 3. Phenomena Under Investigation

#### 3.1. Light-Driven Motion of Photoinjected Carriers

The orthogonal pump-probe geometry largely decouples the transport of carriers from their creation. The electric current induced by the probe pulse in the direction of its electric field decays after the probe pulse ends. The decay rate provides a direct measure of momentum relaxation induced by elastic scattering off the disordered lattice. This is the **primary focus** of the project.

Key questions include:
- How does the momentum relaxation rate depend on disorder strength?
- How does disorder influence the effective mass of the electron wavepacket and the dependence of the effective mass on the reciprocal-space displacement by the probe pulse?
- Can a generalized Drude model with time-dependent effective mass describe the transport across different disorder regimes?

Initial results suggest disorder modifies transport primarily by altering the effective mass landscape and creating more uniform conduction band population, rather than introducing qualitatively new physics.

#### 3.2. Free-Induction Decay (Dephasing)

After photoinjection, coherent electron-hole oscillations (dipole response) decay due to loss of phase coherence. Disorder accelerates this dephasing through inhomogeneous broadening and enhanced scattering. The project investigates whether disorder leaves distinct qualitative signatures in these dipole oscillations beyond generic decay.

**Status:** This aspect remains under investigation. Time-frequency analysis methods have yielded inconclusive results to date. A decision on whether this phenomenon can be meaningfully characterized—and thus whether it will be included in the final thesis—is imminent.

#### 3.3. Photoinjection Dynamics

The project initially aimed to investigate how disorder affects nonlinear carrier generation mechanisms (multiphoton absorption, tunneling ionization). Disorder can modify photoinjection by breaking $k$-selection rules, introducing localized states, and altering the density of states.

**Status:** Initial time-domain investigations revealed no significant qualitative differences between perfect crystals and disordered media in the photoinjection process itself. The dominant disorder signatures appear in the *transport* of carriers after injection, not in their generation. Energy-domain analysis of the photoinjected carrier distribution may yet reveal subtle disorder effects, but this is not the current focus given the tight thesis deadline.

### 4. Computational Approach

**Primary Tool:** Real-time TDDFT using SALMON (Scalable Ab-initio Light-Matter simulator for Optics and Nanoscience), an open-source code that propagates the time-dependent Kohn-Sham equations under the influence of intense electromagnetic fields.

**Disorder Implementation:**
- **Weakly disordered crystals:** Random displacement of atoms from ideal lattice positions, characterized by root-mean-square (RMS) displacement
- **Amorphous silicon:** Atomic configurations generated from structural databases or classical molecular dynamics, characterized by metrics such as Smooth Overlap of Atomic Positions (SOAP)

**Key Methodological Requirements:**
- **Supercell size:** Must be large enough to capture relevant localization length scales and minimize artificial periodicity effects (typically 2×2×2 to 3×3×3 unit cells for silicon)
- **Ensemble size:** Multiple independent disorder realizations (typically 10-20) to obtain converged ensemble-averaged observables for dephasing; no ensemble averaging is necessary for transport dynamics
- **k-point sampling:** For perfect crystals, dense k-grids (8×8×8 or finer) in a single unit cell are required; for disordered crystals, supercells are necessary, which reduces the size of the Brillouin zone and reduces the number of nodes in the k-grid, preserving the grid density

**Analysis Methods:**
- Extraction of macroscopic electric current and its decomposition into pump, probe, and cross-term contributions
- Fitting to phenomenological models (generalized Drude model with the effective mass that depends on the wave packet's average position in reciprocal space)
- Density of states (DOS) analysis from Kohn-Sham orbital energies
- Spectral and time-frequency analysis of dipole oscillations

## Current Research Status and Expected Deliverables

The project has achieved significant methodological success with validation of the orthogonal pump-probe protocol and ensemble averaging approach. Key accomplishments include:

- **Momentum relaxation observed:** Ensemble-averaged simulations successfully capture current decay after the probe pulse, with relaxation rates that systematically increase with disorder strength
- **Phenomenological model developed:** A generalized Drude model with time-dependent effective mass ($m^* \propto A^4$, where $A$ is the probe vector potential) provides excellent fits to TDDFT data across disorder regimes from weakly perturbed crystals to fully amorphous silicon

**Remaining work** (as of January 2026, with thesis submission deadline March 1, 2026):
- Quantitative characterization of how model parameters ($\gamma$, $m^*$, $\alpha$) depend on disorder metrics
- Final decision on free-induction decay analysis (include or exclude from thesis)
- Investigation of whether disorder affects energy distribution of photoinjected carriers

The thesis narrative will center on the orthogonal pump-probe methodology and its revelation that **disorder modifies ultrafast carrier transport primarily via effective mass renormalization and momentum relaxation, with the physics captured by a generalized Drude model showing continuous evolution from weakly disordered to fully amorphous structures**.
