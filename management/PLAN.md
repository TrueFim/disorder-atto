# Project Plan: Optical attosecond spectroscopy of disordered media

**Last Sync:** 2026-02-02

## Status: 🟢 (On Track – Final Push)
*Rationale:* The coherence time metric is finalized: **Mandel's coherence time** (power-equivalent width of the analytic envelope of the autocorrelation). Implemented and validated in `Fourier_analysis.py`. Combined with the validated Drude model for transport, all methodological tools are now in place. The deadline (2026-03-01) is ~4 weeks away. Remaining work is execution: tabulate $\tau_\mathrm{coh}$ across disorder cases, correlate with $\gamma$, and write.

## Active Workflows

### Current Path: Dual-Track Physics Extraction (Transport + FID)
*Rationale:* The orthogonal pump-probe protocol delivers two distinct observables: (1) the slowly-varying current component revealing momentum relaxation and effective mass physics, and (2) the fast dipole oscillations revealing dephasing and spectral restructuring. Both show systematic disorder dependence. The thesis will argue that disorder modifies ultrafast dynamics through a unified mechanism—scattering off the disordered potential—manifesting differently in intraband (transport) and interband (coherence) channels.

### Immediate Next Steps (Priority Order)
1.  **[DONE]** ~~Dipole Oscillations (FID) Decision Point~~ → **Included in thesis.**
2.  **[IN PROGRESS]** **FID Quantification:** Extract coherence times $\tau_\mathrm{coh}$ from autocorrelation analysis.
    - **Metric finalized:** Mandel's coherence time — the power-equivalent width of the normalized analytic envelope of the autocorrelation: $\tau_\mathrm{coh} = 2 \int_0^{\infty} |\mathcal{R}(\tau)/\mathcal{R}(0)|^2\, d\tau$, where $\mathcal{R}(\tau) = R(\tau) + i\mathcal{H}\{R(\tau)\}$. Validated analytically: gives $\tau_\mathrm{coh} = 1/\gamma$ for exponential decay. Progress report: `progress/2026-02-02 coherence time/`.
    - **Implementation done:** `Fourier_analysis.py` computes $\tau_\mathrm{coh}$ for all cases using `scipy.signal.envelope` + `scipy.integrate.trapezoid`. Values displayed on autocorrelation envelope plots.
    - **Remaining:** (a) Tabulate $\tau_\mathrm{coh}$ systematically for all disorder cases. (b) Compute spectral width $\sigma_\omega$ as consistency check ($\tau_\mathrm{coh} \sim 1/\sigma_\omega$). (c) Investigate the 1.4d10_8d12 anomaly (broad spectrum). (d) Analyze pump-only $J_z(t)$ to test for probe-induced dephasing. (e) Note qualitatively that periodic lattice shows revivals (discrete-spectrum beating) while disordered cases show monotonic decay (true dephasing).
3.  **Correlate FID and Transport:** Plot $\tau_\mathrm{coh}$ vs. $\gamma$ to test whether coherence time correlates with momentum relaxation rate. If a clear relationship exists, this unifies the two phenomena.
4.  **Disorder Dependency Plots:** Trofim to systematically plot $\gamma(s)$, $m^*(s)$, $\alpha(s)$, and $\tau_\mathrm{coh}(s)$ where $s$ quantifies disorder (RMS atomic displacement or SOAP metric).
5.  **DOS-FID Link:** Check whether the observed spectral peaks in FID can be matched to features in DOS plots. Can DOS broadening predict $\tau_\mathrm{coh}$?
6.  **Carrier Concentration Extraction:** Extract carrier concentrations to decouple effective mass from carrier density.
7.  **Energy Distribution Analysis:** Investigate whether disorder alters the energy distribution of photoinjected carriers. Lower priority now that FID is confirmed, but still valuable for understanding spectral red shift.

## Strategic Context

### Core Hypothesis (Revised)
Disorder influences femtosecond-scale nonlinear optical response through **scattering off the disordered potential**, which manifests in two complementary channels:

1. **Transport (intraband):** Momentum relaxation captured by Drude model with $\Delta m^*(t) \propto A^4(t)$. Disorder increases $\gamma$ and reduces effective mass variation (more uniform band population).

2. **Coherence (interband):** Dephasing of dipole oscillations. Disorder accelerates decay, red-shifts the spectrum, and restructures discrete crystal peaks into a single low-frequency feature. The spectral phase flattens, suggesting more synchronized oscillators.

Both effects arise from the same underlying physics but probe different aspects: intraband scattering vs. interband coherence loss. A quantitative correlation between $\tau_\mathrm{coh}$ and $\gamma$ would strongly support this unified picture.

### Key Uncertainties
*   **$\tau_\mathrm{coh}$–$\gamma$ Correlation (High priority):** Do coherence time and momentum relaxation rate scale together? If yes, the thesis has a unifying result. If no, there's interesting physics to explain.
*   **Probe-Induced Dephasing (Open):** Does the probe pulse accelerate dephasing beyond intrinsic disorder effects? Pump-only $J_z(t)$ analysis will resolve this.
*   **Anomalous High-Frequency Emergence (Open):** The 1.4d10_8d12 case shows delayed appearance of high-frequency spectral components. If physical, this is a potential discovery (probe-induced excited-state absorption?). Trofim investigating.
*   **Microscopic Origin (Open):** Can DOS broadening quantitatively predict $\tau_\mathrm{coh}$ and $\alpha$? Jankousky et al. (Nature Physics 2026) provides the template: $\mu = |e|\hbar/(2\Delta E \cdot m^*)$.

### Success Criteria for Thesis Completion
By **mid-February 2026**, Trofim must have:
1. Quantitative disorder-dependence plots for all model parameters ($\gamma$, $m^*$, $\alpha$, $\tau_\mathrm{coh}$)
2. $\tau_\mathrm{coh}$ vs. $\gamma$ correlation plot
3. Pump-only $J_z(t)$ analysis for comparison with pump-probe $J_x(t)$
4. Draft thesis outline with figures for both transport and FID chapters

**Thesis narrative (revised):** "Orthogonal pump-probe TDDFT reveals that disorder modifies ultrafast dynamics through scattering, captured by a generalized Drude model for transport and accelerated dephasing for coherence. Both phenomena show systematic disorder dependence, with spectral signatures (red shift, peak restructuring) linked to DOS changes."
