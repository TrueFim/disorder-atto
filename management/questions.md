# Research questions for project "Optical attosecond spectroscopy of disordered media"

## Must know

### How does disorder affect the light-driven motion of electron wavepackets prepared by femtosecond photoinjection?
Is the electric current after the pump pulse described by the Drude model? If not, which of the following is essential: time dependence of the momentum relaxation time, breakdown of the Markov approximation, electron localization, or carrier trapping? How does the momentum-relaxation rate depend on the degree of disorder?  What does the rate correlate with: the linear absorption, the wave packet's kinetic energy, it's position in the reciprocal space, its spatial spread, or something else?

### How does disorder affect free-induction decay (FID) after femtosecond nonlinear photoinjection?
How does the decay of dipole oscillations in a perfect crystal differ from that in a disordered solid? Is FID in a static disordered potential governed solely by linear inhomogeneous broadening, where the decay of the macroscopic polarization $\mathbf{P}(t)$ arises purely from the destructive interference of independent dipoles oscillating at different, statically distributed frequencies? Does the Coulomb many-body scattering matter? Does the dependence on the supercell size suggest that electron localization matters? What are the most appropriate ways to visualize and characterize the effect of disorder on FID?

## Would be good to know

### How does disorder affect two-photon transitions driven by an intense, few-cycle laser pulse?
Is the energy distribution of carriers photoinjected in a disordered semiconductor via multiphoton absorption just a broadened distribution of carriers photoinjected in a perfect crystal? Does disorder favor excited-state absorption within a few-cycle laser pulse (e.g., due to carrier scattering in the presence of a strong electric field)? What is the role of localized electron states? How does electron scattering affect the sub-optical-cycle dynamics of multiphoton transitions?

## May wish to know someday

- Compared to the effect of quenched disorder, how does lattice motion manifest itself in free-induction decay after ultrafast photoinjection?

- To what extent can electron-phonon interaction during a few-femtosecond time interval be mimicked by disorder?

- Is there a qualitative difference between direct and indirect semiconductors in the context of ultrafast phenomena in disordered solids?

- How does disorder influence the Franz-Keldysh effect?

- How does the free-induction decay depend on the carrier-envelope phase (CEP) of the pulse? Is the decay rate sensitive to the CEP? Is there a qualitative difference between single-photon and two-photon photoinjection? 

## Answered questions

### In 3D TDDFT simulations with a linearly polarized pulse, is there a big difference whether we take a supercell that is a big cube, a thin square, or a long rectangular parallelepiped?
Answer: Yes.

## Obsolete questions

### What can we learn about local electric fields by comparing TDDFT and TDSE simulations?
When the exchange-correlation potential is frozen, only the fast oscillations in the electric current change--the rest almost doesn't change. All in all, the effect looks minor, and it gets weaker as the supercell size increases.

### Does disorder fundamentally require abandoning the strict distinction between the interband and intraband currents?
While the formal distinction between interband (excitation) and intraband (transport) currents is foundational for crystals, its breakdown in disordered systems due to the loss of Bloch states presents a profound conceptual and analytical challenge for interpreting ultrafast, strong-field light-matter interactions. The core puzzle is whether, despite this formal invalidity, the physical phenomena of carrier generation and subsequent real-space acceleration can still be meaningfully separated and analyzed in real-time within disordered media. Resolving this ambiguity is critical; it will not only provide a fundamentally new physical insight into how light energy drives carrier dynamics in complex materials but also establish a robust theoretical framework essential for developing predictive *ab initio* models and correctly interpreting attosecond optical signals, thereby advancing materials design and novel optoelectronic applications.