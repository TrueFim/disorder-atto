# Project Plan: Optical attosecond spectroscopy of disordered media

**Last Sync:** 2026-01-21

## Status: 🟢 (On Track – Final Push)
*Rationale:* The FID decision is resolved: **dipole oscillations will be included in the thesis**. Vlad's analysis revealed clear, interpretable disorder signatures—accelerated dephasing, spectral red shift, peak restructuring—that go beyond trivial ensemble averaging. Combined with the validated Drude model for transport, the thesis now has two complementary narratives. The deadline (2026-03-01) is ~5.5 weeks away. Risk level reduced from Yellow to Green because the scope is now fixed and the remaining work is quantitative refinement rather than exploratory.

## Active Workflows

### Current Path: Dual-Track Physics Extraction (Transport + FID)
*Rationale:* The orthogonal pump-probe protocol delivers two distinct observables: (1) the slowly-varying current component revealing momentum relaxation and effective mass physics, and (2) the fast dipole oscillations revealing dephasing and spectral restructuring. Both show systematic disorder dependence. The thesis will argue that disorder modifies ultrafast dynamics through a unified mechanism—scattering off the disordered potential—manifesting differently in intraband (transport) and interband (coherence) channels.

### Immediate Next Steps (Priority Order)
1.  **[DONE]** ~~Dipole Oscillations (FID) Decision Point~~ → **Included in thesis.**
2.  **[IN PROGRESS]** **FID Quantification:** Extract dephasing times $T_2$ from envelope fits. Investigate the 1.4d10_8d12 anomaly (delayed high-frequency emergence). Analyze pump-only $J_z(t)$ to test for probe-induced dephasing.
3.  **Correlate FID and Transport:** Plot $T_2$ vs. $\gamma$ to test whether dephasing time correlates with momentum relaxation rate. If a clear relationship exists, this unifies the two phenomena.
4.  **Disorder Dependency Plots:** Trofim to systematically plot $\gamma(s)$, $m^*(s)$, $\alpha(s)$, and now $T_2(s)$ where $s$ quantifies disorder (RMS atomic displacement or SOAP metric).
5.  **DOS-FID Link:** Check whether the observed spectral peaks in FID can be matched to features in DOS plots. Can DOS broadening predict $T_2$?
6.  **Carrier Concentration Extraction:** Extract carrier concentrations to decouple effective mass from carrier density.
7.  **Energy Distribution Analysis:** Investigate whether disorder alters the energy distribution of photoinjected carriers. Lower priority now that FID is confirmed, but still valuable for understanding spectral red shift.

## Strategic Context

### Core Hypothesis (Revised)
Disorder influences femtosecond-scale nonlinear optical response through **scattering off the disordered potential**, which manifests in two complementary channels:

1. **Transport (intraband):** Momentum relaxation captured by Drude model with $m^*(t) \propto A^4(t)$. Disorder increases $\gamma$ and reduces effective mass variation (more uniform band population).

2. **Coherence (interband):** Dephasing of dipole oscillations. Disorder accelerates decay, red-shifts the spectrum, and restructures discrete crystal peaks into a single low-frequency feature. The spectral phase flattens, suggesting more synchronized oscillators.

Both effects arise from the same underlying physics but probe different aspects: intraband scattering vs. interband coherence loss. A quantitative correlation between $T_2$ and $\gamma$ would strongly support this unified picture.

### Key Uncertainties
*   **$T_2$–$\gamma$ Correlation (High priority):** Do dephasing and momentum relaxation rates scale together? If yes, the thesis has a unifying result. If no, there's interesting physics to explain.
*   **Probe-Induced Dephasing (Open):** Does the probe pulse accelerate dephasing beyond intrinsic disorder effects? Pump-only $J_z(t)$ analysis will resolve this.
*   **Anomalous High-Frequency Emergence (Open):** The 1.4d10_8d12 case shows delayed appearance of high-frequency spectral components. If physical, this is a potential discovery (probe-induced excited-state absorption?). Trofim investigating.
*   **Microscopic Origin (Open):** Can DOS broadening quantitatively predict $T_2$ and $\alpha$? Jankousky et al. (Nature Physics 2026) provides the template: $\mu = |e|\hbar/(2\Delta E \cdot m^*)$.

### Success Criteria for Thesis Completion
By **mid-February 2026**, Trofim must have:
1. Quantitative disorder-dependence plots for all model parameters ($\gamma$, $m^*$, $\alpha$, $T_2$)
2. $T_2$ vs. $\gamma$ correlation plot
3. Pump-only $J_z(t)$ analysis for comparison with pump-probe $J_x(t)$
4. Draft thesis outline with figures for both transport and FID chapters

**Thesis narrative (revised):** "Orthogonal pump-probe TDDFT reveals that disorder modifies ultrafast dynamics through scattering, captured by a generalized Drude model for transport and accelerated dephasing for coherence. Both phenomena show systematic disorder dependence, with spectral signatures (red shift, peak restructuring) linked to DOS changes."
