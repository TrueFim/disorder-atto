import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Fourier analysis of the free-induction decay

    For the ideal crystal and the case of randomly displaced atoms, the simulations were performed for the same set of peak laser intensities. These results can be directly compared with each other. For the amorphous silicon, all the data files are for the same peak laser intensity, but they represent three different degrees of disorder.
    """)
    return


@app.cell
def _():
    subtract_non_FID_signal = True
    pump_pulse_t0 = 6.0 # (fs) the peak of the pump pulse
    return pump_pulse_t0, subtract_non_FID_signal


@app.cell
def _(mo):
    mo.md(r"""
    ## Load the data and perform the Fourier transform
    """)
    return


@app.cell
def _():
    import marimo as mo
    import attoworld as aw
    from attoworld.personal.vlad import soft_window, Fourier_transform
    import numpy as np
    import scipy
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    au = aw.numeric.AtomicUnits()
    return (
        Fourier_transform,
        PdfPages,
        au,
        aw,
        mo,
        np,
        os,
        plt,
        scipy,
        soft_window,
    )


@app.cell
def _(plt):
    # font sizes for plots
    plt.rcParams['axes.titlesize'] = 14     # Title of the subplots
    plt.rcParams['axes.labelsize'] = 12     # x and y labels
    plt.rcParams['xtick.labelsize'] = 11    # x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 11    # y-axis tick labels
    plt.rcParams['legend.fontsize'] = 11    # Legend font size
    return


@app.cell
def _():
    # load all the data files
    from load_csv_tree import load_csv_data
    all_data = load_csv_data(".")
    return (all_data,)


@app.cell
def _(all_data, np, pump_pulse_t0, scipy, subtract_non_FID_signal):
    # select the relevant data and fix the time axis
    def select_relevant_data_and_fix_time_axis(all_data, t0, subtract_non_FID_signal):
    	all_relevant_data = {}
    	for dir1 in all_data:
    		relevant_data = {}
    		for dir2 in all_data[dir1]['data']:
    			data1 = all_data[dir1]['data'][dir2]['full_t']['data1'].copy()
    			data2 = all_data[dir1]['data'][dir2]['cut_t']['data2'].copy()
    			time_mismatch = data1[-1,0] - data2[-1,0]
    			data1[:, 0] -= t0
    			data2[:, 0] += time_mismatch - t0
    			if subtract_non_FID_signal:
    				# extrapolate the Drude fit
    				Drude_t = data2[:, 0]
    				Drude_fit = data2[:, 2]
    				i1 = np.flatnonzero(Drude_fit >= 0.5 * np.max(Drude_fit))[0]
    				t1 = Drude_t[i1]
    				y1 = Drude_fit[i1]
    				dydt1 = (Drude_fit[i1+1] - Drude_fit[i1]) / (Drude_t[i1+1] - Drude_t[i1])
    				a = (dydt1 * t1 - 2 * y1) / t1**3
    				b = (3 * y1 - dydt1 * t1) / t1**2
    				t_grid = data1[:, 0]
    				non_FID_component = np.zeros(len(t_grid))
    				s = np.logical_and(t_grid>0, t_grid<t1)
    				non_FID_component[s] = (a * t_grid[s] + b) * t_grid[s]**2
    				interpolator = scipy.interpolate.CubicSpline(Drude_t, Drude_fit, extrapolate=True)
    				s = (t_grid >= t1)
    				non_FID_component[s] = interpolator(t_grid[s])
    				# # TEST BEGIN
    				# plt.plot(t_grid, data1[:, 1])
    				# plt.plot(t_grid, non_FID_component)
    				# #plt.plot(data1[:, 0], data1[:, 1])
    				# #plt.plot(data2[:, 0], data2[:, 1])
    				# #plt.plot(data2[:, 0], data2[:, 2])
    				# plt.show()
    				# # TEST END
    				data1[:, 1] -= non_FID_component
    			relevant_data[dir2] = data1
    		all_relevant_data[dir1] = relevant_data
    	return all_relevant_data

    FID_data_time_domain = select_relevant_data_and_fix_time_axis(all_data, pump_pulse_t0, subtract_non_FID_signal)
    return (FID_data_time_domain,)


@app.cell
def _(FID_data_time_domain, Fourier_transform, au, np, soft_window):
    # Fourer transform all the relevant data
    def Fourier_transform_relevant_data(all_relevant_data, omega_array, softening_time):
        Fourier_transformed_data = {}
        for dir1 in all_relevant_data:
            Fourier_transformed_data[dir1] = {}
            for dir2 in all_relevant_data[dir1]:
                data = all_relevant_data[dir1][dir2]
                t = data[:, 0]
                window = soft_window(t, t[-1]-softening_time, t[-1])
                # # TEST BEGIN
                # plt.plot(t, data[:, 1])
                # plt.plot(t, data[:, 1] * window)
                # plt.show()
                # # TEST END
                transformed_data = Fourier_transform(t, data[:, 1] * window, omega_array, is_periodic=False, pulse_center_times=0.0)
                Fourier_transformed_data[dir1][dir2] = transformed_data
        return Fourier_transformed_data

    omega_array = np.linspace(1., 8.0, 500) / au.eV / au.fs
    FID_data_frequency_domain = Fourier_transform_relevant_data(FID_data_time_domain, omega_array, softening_time=10.0)
    return FID_data_frequency_domain, omega_array


@app.cell
def _(mo):
    mo.md(r"""
    ## Compare the simulations with periodic and disordered lattices
    """)
    return


@app.cell
def _(
    FID_data_frequency_domain,
    FID_data_time_domain,
    PdfPages,
    au,
    np,
    omega_array,
    plt,
):
    def perform_visualization1(time_domain_data, omega_array, frequency_domain_data):
        with PdfPages('cryst_vs_displ.pdf') as pdf:
            for dir in sorted(time_domain_data['cryst']):
                fig, axs = plt.subplots(3, 1, figsize=(8.27*0.9, 11.69*0.9))
                fig.suptitle(dir, fontsize=16)
                # time-domain plot
                ax = axs[0]
                X = time_domain_data['cryst'][dir][:,0]
                Y = time_domain_data['cryst'][dir][:,1]
                ax.plot(X, Y, label='periodic lattice')
                X = time_domain_data['displ'][dir][:,0]
                Y = time_domain_data['displ'][dir][:,1]
                ax.plot(X, Y, label='random displacements')
                ax.set_xlabel('time (fs)')
                ax.set_ylabel(r'$J_x(t)$ (atomic units)')
                ax.set_title('Electric current after subtracting wave packet motion')
                ax.legend()
                ax.set_xlim(X[0], X[-1])
                # spectral intensities
                ax = axs[1]
                X = omega_array * au.fs * au.eV
                Y = np.abs(frequency_domain_data['cryst'][dir])**2
                Y /= np.max(Y)
                i1 = np.argmax(Y) # we'll need it to unwrap the spectral phase
                ax.plot(X, Y, label='periodic lattice')
                Y = np.abs(frequency_domain_data['displ'][dir])**2
                Y /= np.max(Y)
                i2 = np.argmax(Y) # we'll need it to unwrap the spectral phase
                ax.plot(X, Y, label='random displacements')
                ax.set_xlabel(r'$\hbar\omega$ (eV)')
                ax.set_title('Normalized spectral intensity')
                ax.legend()
                ax.set_xlim(X[0], X[-1])
                # spectral phases
                ax = axs[2]
                # X = omega_array * au.fs * au.eV
                Y = np.angle(frequency_domain_data['cryst'][dir])
                Y[i1:] = np.unwrap(Y[i1:])
                Y[i1::-1] = np.unwrap(Y[i1::-1])
                Y_min = np.min(Y)
                Y_max = np.max(Y)
                ax.plot(X, Y, label='periodic lattice')
                Y = np.angle(frequency_domain_data['displ'][dir])
                Y[i2:] = np.unwrap(Y[i2:])
                Y[i2::-1] = np.unwrap(Y[i2::-1])
                Y_min = min(Y_min, np.min(Y))
                Y_max = max(Y_max, np.max(Y))
                ax.plot(X, Y, label='random displacements')
                ax.set_xlabel(r'$\hbar\omega$ (eV)')
                ax.set_ylabel('radians')
                ax.set_title('Spectral phase')
                ax.set_xlim(X[0], X[-1])
                ax.set_ylim(max(-20, Y_min), min(20, Y_max))
                ax.legend()
                # # Adjust margins
                # plt.subplots_adjust(
                #     left=0.1,   # left margin
                #     right=0.9,  # right margin
                #     bottom=0.1, # bottom margin
                #     top=0.9,    # top margin
                #     wspace=0.3
                # )
                plt.tight_layout()
                pdf.savefig()

    perform_visualization1(FID_data_time_domain, omega_array, FID_data_frequency_domain)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Compare the simulations with amorphous silicon
    """)
    return


@app.cell
def _(
    FID_data_frequency_domain,
    FID_data_time_domain,
    au,
    np,
    omega_array,
    plt,
):
    def perform_visualization2(time_domain_data, omega_array, frequency_domain_data):
        fig, axs = plt.subplots(3, 1, figsize=(8.27*0.9, 11.69*0.9))
        axs[0].set_title('Electric current after subtracting wave packet motion')
        axs[0].set_xlabel('time (fs)')
        axs[0].set_ylabel(r'$J_x(t)$ (atomic units)')
        axs[1].set_title('Normalized spectral intensity')
        axs[1].set_xlabel(r'$\hbar\omega$ (eV)')
        axs[2].set_title('Spectral phase')
        axs[2].set_xlabel(r'$\hbar\omega$ (eV)')
        axs[2].set_ylabel('radians')
        Y_min = 0 # for the spectral phase
        Y_max = 0 # for the spectral phase
        for dir in sorted(time_domain_data['amorph']):
            # time-domain plot
            ax = axs[0]
            X = time_domain_data['amorph'][dir][:,0]
            Y = time_domain_data['amorph'][dir][:,1]
            ax.plot(X, Y, label=dir)
            ax.set_xlim(X[0], X[-1])
            # spectral intensities
            ax = axs[1]
            X = omega_array * au.fs * au.eV
            Y = np.abs(frequency_domain_data['amorph'][dir])**2
            Y /= np.max(Y)
            i1 = np.argmax(Y) # we'll need it to unwrap the spectral phase
            ax.plot(X, Y, label=dir)
            ax.set_xlim(X[0], X[-1])
            # spectral phases
            ax = axs[2]
            # X = omega_array * au.fs * au.eV
            Y = np.angle(frequency_domain_data['amorph'][dir])
            Y[i1:] = np.unwrap(Y[i1:])
            Y[i1::-1] = np.unwrap(Y[i1::-1])
            Y_min = min(Y_min, np.min(Y))
            Y_max = max(Y_max, np.max(Y))
            ax.plot(X, Y, label=dir)
            ax.set_xlim(X[0], X[-1])
        axs[2].set_ylim(max(-20, Y_min), min(20, Y_max))
        for i in range(3):
            axs[i].legend()
        plt.tight_layout()
        plt.savefig('amorph.pdf', format='pdf')

    perform_visualization2(FID_data_time_domain, omega_array, FID_data_frequency_domain)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Time-frequency analysis
    """)
    return


@app.cell
def _(FID_data_time_domain, aw, np, omega_array, os, plt):
    def perform_time_frequency_analysis(time_domain_data):
        for dir1 in time_domain_data:
            for dir2 in time_domain_data[dir1]:
                X = 1e-15 * np.ascontiguousarray(time_domain_data[dir1][dir2][:,0])
                Y = np.ascontiguousarray(time_domain_data[dir1][dir2][:,1])
                waveform = aw.data.Waveform(time=X, wave=Y, dt=None)
                # spectrogram = aw.wave.stft(waveform, nperseg=1024*4)
                spectrogram = aw.wave.cwt(waveform)
                fig, ax = plt.subplots()
                ax.set_xlim(-5., 20.)  # fs
                f_max_THz = 1000 * omega_array[-1] / (2 * np.pi)
                ax.set_ylim(200, f_max_THz)
                ax.set_title(f"{dir1}, {dir2}")
                spectrogram.plot(ax, take_sqrt=True)
                plt.tight_layout()
                path = os.path.join(dir1, 'ims', f"time_frequency_{dir2}.png")
                plt.savefig(path, dpi=300, format='png')
                plt.show()

    perform_time_frequency_analysis(FID_data_time_domain)
    return


if __name__ == "__main__":
    app.run()
