# Project Plan: Optical attosecond spectroscopy of disordered media

**Last Sync:** 2026-02-18

## Status: 🟡 (Caution – Timeline Slipping)
*Rationale:* Thesis deadline is **2026-03-01** (10 days). The mid-February milestone for disorder-dependence plots was missed — Trofim has not yet produced $\gamma(s)$, $m^*(s)$, $\alpha(s)$, $\tau_\mathrm{coh}(s)$ plots. Additionally, the key expected result — a $\tau_\mathrm{coh}$–$\gamma$ correlation — is **not confirmed**: preliminary data shows Mandel's $\tau_\mathrm{coh}$ (from $J_z$, pump-only) does not correlate with Drude's $\gamma$ (from $J_x$, pump-probe). An alternative metric is under investigation. The probe-induced dephasing signal is observed but not yet compared to the perfect crystal case. The next ~5 days are critical for resolving the correlation question and producing plots.

## Active Workflows

### Current Path: Dual-Track Physics Extraction (Transport + FID)
*Rationale:* Transport (Drude model) is well-established. FID phenomenology is documented. The key open question is whether a quantitative link between transport and coherence exists.

### Immediate Next Steps (Priority Order)
1.  **[URGENT – Trofim]** **Produce disorder-dependence plots:** $\gamma(s)$, $m^*(s)$, $\alpha(s)$, $\tau_\mathrm{coh}(s)$ for all available cases. These are critical for the thesis; must be done immediately.
2.  **[IN PROGRESS]** **Resolve the $\tau_\mathrm{coh}$–$\gamma$ correlation question:**
    - Mandel's $\tau_\mathrm{coh}$ from $J_x(t)$ (pump-probe) shows **no correlation** with $\gamma$ (preliminary).
    - Trofim's alternative: exponential fit to local maxima of $J_z(t)$ (pump-only) produces values close to $\gamma$. Methodologically shaky (manual outlier exclusion; inapplicable to perfect crystal).
    - **Next step (Vlad):** Produce new version of `ensemble_dephasing.ipynb` replacing time-frequency analysis with autocorrelation-based analysis.
    - **Next step (Trofim):** If the autocorrelations in `ensemble_dephasing.ipynb` look like steadily decaying signals, fit an exponential to the maxima of the autcorrelation function. If this produces prominent results, apply the method to the **autocorrelation of $J_z(t)$** instead of the raw signal. 
3.  **[URGENT – Trofim]** **Probe-induced dephasing comparison:** Compare $J_z(t)$ dephasing in pump-probe vs. pump-only for both disordered AND perfect crystal cases. If disorder induces probe-induced dephasing, this is a major finding. If not, then it's just one of the many interesting observations.
4.  **DOS-FID Link:** Check whether spectral peaks in FID match DOS features. Can DOS broadening quantitatively predict $\tau_\mathrm{coh}$?
5.  **Carrier Concentration Extraction:** Extract carrier concentrations to decouple effective mass from carrier density.
6.  **Energy Distribution Analysis:** Lowest priority; defer if time is insufficient.

## Strategic Context

### Core Hypothesis (Under Pressure)
Disorder influences femtosecond-scale nonlinear optical response through **scattering off the disordered potential**, which manifests in two complementary channels:

1. **Transport (intraband):** Momentum relaxation captured by Drude model with $\Delta m^*(t) = \alpha A^4(t)$. Disorder increases $\gamma$ and reduces effective mass variation (more uniform band population). *(Well-established.)*

2. **Coherence (interband):** Dephasing of dipole oscillations along the pump direction. Disorder accelerates decay, red-shifts the spectrum, restructures discrete crystal peaks into a single low-frequency feature. *(Well-established qualitatively.)*

**Unifying link under question:** A quantitative $\tau_\mathrm{coh}$–$\gamma$ correlation would strongly support the unified picture, but preliminary results with Mandel's metric are negative. A more systematic investiation of FID ib both $J_x(t)$ and $J_z(t)$ is necessary to make a final conclusion.

**Contingency:** If no quantitative correlation is found, the thesis narrative shifts to: "Disorder modifies both transport and coherence through the same scattering mechanism, but the two channels probe different physical regimes and cannot be simply related via a single parameter."

### Key Uncertainties
*   **$\tau_\mathrm{coh}$–$\gamma$ Correlation (Critical):** Mandel's metric (from $J_x$, pump-probe) shows no correlation. Alternative metric (fit to $J_z$ autocorrelation maxima, pump-only) under investigation. Resolution needed in days.
*   **Probe-Induced Dephasing (Partially observed):** Probe significantly accelerates $J_z$ dephasing in disordered case. Crystal comparison pending. If the effect is disorder-specific, it becomes a key finding.
*   **Conceptual distinction from Purschke et al.:** TDDFT FID = unitary inhomogeneous broadening. Purschke et al. = phenomenological spatial dissipation $\Gamma_\mathrm{RS}$. These are distinct mechanisms; the pump-only vs. pump-probe comparison discriminates between them.
*   **Anomalous High-Frequency Emergence (Open):** The 1.4d10_8d12 case shows delayed high-frequency components. If physical, this may indicate probe-induced excited-state absorption.
*   **Microscopic Origin (Open):** Can DOS broadening quantitatively predict $\tau_\mathrm{coh}$ and $\alpha$? Template: Jankousky et al. $\mu = |e|\hbar/(2\Delta E \cdot m^*)$.

### Success Criteria for Thesis Completion
By **2026-03-01** (thesis deadline), the required deliverables are:
1. Quantitative disorder-dependence plots: $\gamma(s)$, $m^*(s)$, $\alpha(s)$, $\tau_\mathrm{coh}(s)$ *(not yet done — critical)*
2. Resolution of the $\tau_\mathrm{coh}$–$\gamma$ correlation question (positive or negative, with physical interpretation)
3. Pump-only $J_z(t)$ vs. pump-probe comparison for crystal and disordered cases
4. Complete thesis draft with figures for both transport and FID chapters

**Thesis narrative (revised):** "Orthogonal pump-probe TDDFT reveals that disorder modifies ultrafast dynamics through scattering, captured by a generalized Drude model for transport and accelerated dephasing for coherence. Both phenomena show systematic disorder dependence. Whether transport and coherence dynamics are quantitatively linked is under investigation; if confirmed, this provides a unified description of disorder effects across intraband and interband channels."
