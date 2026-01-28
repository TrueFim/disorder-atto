# Research findings for project "Optical attosecond spectroscopy of disordered media"

## Known for sure
- In the absence of an external electric field, the electric current (in TDDFT simulations) decays exponentially, which means that the relaxation rate is preserved during this process.
- A Drude-like model where the effective mass depends on the instantaneous vector potential (proxy for crystal momentum) provides excellent fits to the TDDFT data for both perturbed crystals and amorphous silicon.
- The variation in effective mass required to fit the current decay decreases as the degree of disorder increases (likely due to more uniform conduction band population).
- Disorder smoothes the peaks and broadens the energy bands in the Density of States (DOS) calculated by SALMON. This has been confirmed with robust extraction methods based on processing Kohn-Sham orbital energies.
- The generalized Drude model with time-dependent effective mass (where $\Delta m^* \propto A^4$ and $A$ is the probe vector potential) successfully describes current dynamics in amorphous silicon, not just weakly disordered crystals.

## Probably true
- The phenomenological parameters of the generalized Drude model ($\gamma$, $m^*$, $\alpha$) exhibit systematic dependencies on the degree and character of disorder, with more uniform conduction band population in highly disordered media reducing the required effective mass variation.
- Disorder accelerates free-induction decay (dephasing of dipole oscillations $J_x(t)$ after pump-probe excitation). The effect is clearly visible in time-domain data.
- Disorder red-shifts the FID spectrum rather than simply broadening it. This may reflect preferential population of lower-energy states enabled by $k$-selection rule breakdown and DOS redistribution.
- For a perfect crystal, the FID spectrum consists of discrete peaks corresponding to specific transitions; disorder merges these into a single dominant low-frequency feature (especially pronounced in amorphous Si).
- In amorphous silicon, the FID spectral width and red shift depend monotonically on the SOAP disorder metric: more disorder → broader spectrum shifted to lower frequencies.

## Seems to be true
- Disorder does not produce significant qualitative differences in the time-domain dynamics of photoinjection (based on initial investigations of the nonlinear work), suggesting that the primary signature of disorder lies in transport and dephasing rather than the injection process itself.
- Disorder minimally affects the early-time excitation dynamics: $J_x(t)$ for a perfect crystal and a disordered solid stay in phase for several femtoseconds when the pump pulse is moderately strong ($3.5 \times 10^9$ W/cm²).
- The spectral phase of FID in disordered media is flatter than in perfect crystals, possibly indicating more synchronized oscillator phases due to homogeneous population.
