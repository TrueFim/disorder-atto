#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[115]:


import numpy as np
import scipy
from scipy import signal
from scipy.signal import hilbert
from scipy import integrate

try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex

import warnings
import re
import matplotlib.image as mpimg
import os
import shutil
import csv
import glob

from PIL import Image
from itertools import product

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from functools import partial

# Полностью отключить все предупреждения
warnings.filterwarnings("ignore")

# os.makedirs("ims", exist_ok=True)
# os.makedirs("ims/current", exist_ok=True)
# os.makedirs("ims/all", exist_ok=True)

# VERY CAREFULLY not to delete other folders!
if os.path.isdir("all_data"):
    shutil.rmtree("all_data")
    print("The all_data folder is deleted and created again")
    print(" ")

os.makedirs("all_data", exist_ok=True)

# os.makedirs("all_data/inputs", exist_ok=True)

os.makedirs("all_data/current_energy", exist_ok=True)
os.makedirs("all_data/current_energy/data", exist_ok=True)
os.makedirs("all_data/current_energy/ims", exist_ok=True)
os.makedirs("all_data/current_energy/ims/all", exist_ok=True)
os.makedirs("all_data/current_energy/ims/energy_current", exist_ok=True)

os.makedirs("all_data/current_energy/ims/all/png", exist_ok=True)
os.makedirs("all_data/current_energy/ims/all/pdf", exist_ok=True)

# os.makedirs("all_data/current_energy/ims/energy_current/png", exist_ok=True)
# os.makedirs("all_data/current_energy/ims/energy_current/pdf", exist_ok=True)



# os.makedirs("all_data/hhg", exist_ok=True)
# os.makedirs("all_data/hhg/data", exist_ok=True)
# os.makedirs("all_data/hhg/ims", exist_ok=True)
# os.makedirs("all_data/hhg/ims/png", exist_ok=True)
# os.makedirs("all_data/hhg/ims/pdf", exist_ok=True)

# os.makedirs("all_data/nonlinear", exist_ok=True)

# os.makedirs("all_data/drude", exist_ok=True)
# os.makedirs("all_data/drude/data", exist_ok=True)
# os.makedirs("all_data/drude/data/current", exist_ok=True)
# os.makedirs("all_data/drude/data/m", exist_ok=True)
# os.makedirs("all_data/drude/data/abc", exist_ok=True)
# os.makedirs("all_data/drude/data/tau", exist_ok=True)
# os.makedirs("all_data/drude/data/n_ex", exist_ok=True)
# os.makedirs("all_data/drude/ims", exist_ok=True)

# os.makedirs("all_data/drude/ims/png", exist_ok=True)
# os.makedirs("all_data/drude/ims/pdf", exist_ok=True)

# os.makedirs("all_data/fid", exist_ok=True)
# os.makedirs("all_data/fid/z", exist_ok=True)
# os.makedirs("all_data/fid/z/data", exist_ok=True)
# os.makedirs("all_data/fid/z/ims", exist_ok=True)
# os.makedirs("all_data/fid/z/ims/png", exist_ok=True)
# os.makedirs("all_data/fid/z/ims/pdf", exist_ok=True)


volume = 160.1 # для Si в A^3

# m0_cut_1 = 10500  # 21 fs: for exp fit
# m0_cut_2 = 6250   # 12.5 fs: for full drude fit (needs to be adjusted in accordance to z_pump energy)
# # 3749 (7.5 fms), 5999 (12 fms)

# m0_cut_3 = 5250 # 10.5 fs: start of integration for Δk analysis. half of probe pulse duration

# In[116]:


plt.rcParams['font.family'] = 'serif'
import matplotlib
# matplotlib.use('Agg') for server (not necessary)
# matplotlib.use("TkAgg")
# print(matplotlib.get_backend())


# In[ ]:





# ## all plots func

# In[116]:


def plot_nonlinear_response_v4(folder, ax, j):


    t_probe, t_pump = get_pulse_durations(folder)

    m0_cut_1 = int(500*t_probe) # duration of probe pulse for exp fit after it
    m0_cut_3 = int(m0_cut_1/2)  # start of integration for Δk analysis. half of probe pulse duration

    m0_cut_2 = m0_cut_3+500 # adjustable. start for drude x-fit (needs to be adjusted in accordance to z_pump energy)

    m0_cut_4 = int(500*(t_probe+t_pump)/2) # pump exp fit, at the end of the pump pulse

    # Загрузка данных, пропуская строки с комментариями (начинаются с #)
    data_both = pd.read_csv(
        folder + '/both_pulses_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # data_pump = pd.read_csv(
    #     folder + '/pump_pulse_rt.data',
    #     comment='#',
    #     delim_whitespace=True,
    #     header=None
    # )

    # data_probe = pd.read_csv(
    #     folder + '/probe_pulse_rt.data',
    #     comment='#',
    #     delim_whitespace=True,
    #     header=None
    # )


    # Назначим читаемые имена колонкам
    data_both.columns = ['time_fs', 'Ac_ext_x', 'Ac_ext_y', 'Ac_ext_z', 'E_ext_x', 'E_ext_y', 'E_ext_z',         
        'Ac_tot_x', 'Ac_tot_y', 'Ac_tot_z', 'E_tot_x', 'E_tot_y', 'E_tot_z',         
        'Jm_x', 'Jm_y', 'Jm_z']

    # data_pump.columns = ['time_fs', 'Ac_ext_x', 'Ac_ext_y', 'Ac_ext_z', 'E_ext_x', 'E_ext_y', 'E_ext_z',         
    #     'Ac_tot_x', 'Ac_tot_y', 'Ac_tot_z', 'E_tot_x', 'E_tot_y', 'E_tot_z',         
    #     'Jm_x', 'Jm_y', 'Jm_z']

    # data_probe.columns = ['time_fs', 'Ac_ext_x', 'Ac_ext_y', 'Ac_ext_z', 'E_ext_x', 'E_ext_y', 'E_ext_z',         
    #     'Ac_tot_x', 'Ac_tot_y', 'Ac_tot_z', 'E_tot_x', 'E_tot_y', 'E_tot_z',         
    #     'Jm_x', 'Jm_y', 'Jm_z']


    t = data_both['time_fs']


    Jm_both_z  =  data_both['Jm_z']
    # Jm_pump_z  =  data_pump['Jm_z']

    Jm_both_x  =  data_both['Jm_x']
    # Jm_pump_x  =  data_pump['Jm_x']
    # Jm_probe_x =  data_probe['Jm_x']

    Jm_both_y  =  data_both['Jm_y']
    # Jm_pump_y  =  data_pump['Jm_y']
    # Jm_probe_y =  data_probe['Jm_y']


    El_f_pump_z = data_both['E_ext_z']
    El_f_probe_x = data_both['E_ext_x']


    # 1:Time[fs] 2:Ac_ext_x[fs*V/Angstrom] 3:Ac_ext_y[fs*V/Angstrom] 4:Ac_ext_z[fs*V/Angstrom]
    #5:E_ext_x[V/Angstrom] 6:E_ext_y[V/Angstrom] 7:E_ext_z[V/Angstrom] 8:Ac_tot_x[fs*V/Angstrom]
    # 9:Ac_tot_y[fs*V/Angstrom] 10:Ac_tot_z[fs*V/Angstrom] 11:E_tot_x[V/Angstrom]
    # 12:E_tot_y[V/Angstrom] 13:E_tot_z[V/Angstrom]  14:Jm_x[1/fs*Angstrom^2] 15:Jm_y[1/fs*Angstrom^2] 16:Jm_z[1/fs*Angstrom^2]


    # Построим графики
    plt.figure(figsize=(10, 7.5))

    # Общий заголовок для всей фигуры
    plt.suptitle(' '.join(re.split('[/_]', folder)), fontsize=14, y=0.99)

    # Первый график: внешнее поле по z
    plt.subplot(3, 2, 1)
    plt.plot(t, El_f_pump_z, label='E_ext_z', color='#2CD311')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [V/Å]')
    plt.title('Pump external electric field z')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()

    # Второй график: внешнее поле по x
    plt.subplot(3, 2, 2)
    plt.plot(t, El_f_probe_x, label='E_ext_z', color='c')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [V/Å]')
    plt.title('Probe external electric field x')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()

    # # Четвертый график: суммарный ток по z
    # plt.subplot(5, 2, 4)
    # plt.plot(t, Jm_pump_z, color='magenta')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    # plt.title('Matter current density along z, only pump')
    # plt.grid(True)

    # plt.tight_layout()






    t_array_cut = np.array(t[m0_cut_4:])[::50]
    j_exp = np.array(Jm_both_z[m0_cut_4:])[::50]
    # (dummy_t, j0, j_final, tau 6.65)


    p0 = [0.02, 0, 20]

    try:
        popt, pcov = curve_fit(drude_exp_fit_v2_pump, t_array_cut, j_exp, p0=p0)
    except RuntimeError:
        print("First fit failed, trying different initial guess...")

        p0_v2 = [0.05, 0, 6.65]
        popt, pcov = curve_fit(drude_exp_fit_v2_pump, t_array_cut, j_exp, p0=p0_v2)


    j0, j_final, tau_pump= popt
    j_fit = drude_exp_fit_v2_pump(0, j0, j_final, tau_pump)

    # ratio = j_final/max(delta_Jm_x)

    # Разностный ток по x
    plt.subplot(3, 2, 4)
    plt.plot(t, Jm_both_z, color='m')
    if tau_pump<30:
        plt.plot(t_array_cut, j_fit, '--', label=rf"$\tau = {tau_pump:.2f}\ \mathrm{{fs}}$", color='r')
        plt.legend()
    else: plt.plot(t_array_cut, j_fit, '--', color='r')
    plt.xlabel('Time [fs]')
    plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    plt.title('Matter current density along z, only pump')
    plt.grid(True)
    plt.tight_layout()








    product_both_z = El_f_pump_z * Jm_both_z * volume
    # Разность по времени
    dt = np.diff(t, prepend=t[0])  # prepend чтобы сохранить размер
    integral_both_z = np.cumsum(product_both_z * dt)

    product_both_x = El_f_probe_x * Jm_both_x * volume
    # Разность по времени
    dt = np.diff(t, prepend=t[0])  # prepend чтобы сохранить размер
    integral_both_x = np.cumsum(product_both_x * dt)

    # product_delta_x = El_f_probe_x * (Jm_both_x - Jm_pump_x - Jm_probe_x) * volume
    # # Разность по времени
    # dt = np.diff(t, prepend=t[0])  # prepend чтобы сохранить размер
    # integral_delta_x = np.cumsum(product_delta_x * dt)



    # Построение графика изменения энергии во времени
    plt.subplot(3, 2, 3)
    # plt.plot(energy_data['Time_fs'], energy_data['Delta_E_eV'], label='ΔE = E_total - E_initial', color='r')
    plt.plot(t, -integral_both_z, label='ΔE = E_total - E_initial', color='r')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [eV]')
    plt.title('Energy transfer per unit cell along z')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()



    # Построение графика изменения энергии во времени
    plt.subplot(3, 2, 5)
    # plt.plot(energy_data['Time_fs'], energy_data['Delta_E_eV'], label='ΔE = E_total - E_initial', color='r')
    plt.plot(t, -integral_both_x, label='ΔE = E_total - E_initial', color='blue')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [eV]')
    plt.title('Energy transfer per unit cell along x')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()


    # # Построение графика изменения энергии во времени
    # plt.subplot(5, 2, 6)
    # # plt.plot(energy_data['Time_fs'], energy_data['Delta_E_eV'], label='ΔE = E_total - E_initial', color='r')
    # plt.plot(t, -integral_delta_x, label='ΔE = E_total - E_initial', color='black')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('E [eV]')
    # plt.title('Δ energy transfer per unit cell along x')
    # plt.grid(True)
    # #plt.legend()
    # plt.tight_layout()



    # # Суммарный ток по x
    # plt.subplot(5, 2, 7)
    # plt.plot(t, Jm_both_x, color='#DC143C')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    # plt.title('Full matter current density along x')
    # plt.grid(True)
    # plt.tight_layout()




    # delta_Jm_x = Jm_both_x - Jm_pump_x - Jm_probe_x


    t_array_cut = np.array(t[m0_cut_1:])[::50]
    j_exp = np.array(Jm_both_x[m0_cut_1:])[::50]
    # (dummy_t, j0, j_final, tau)
    p0 = [0.03, 0.01, 6.65]   # начальные приближения (n, tau)
    popt, pcov = curve_fit(drude_current_v1, t_array_cut, j_exp, p0=p0)
    j0, j_final, tau= popt
    j_fit = drude_current_v1(0, j0, j_final, tau)

    ratio = j_final/max(Jm_both_x)

    # Разностный ток по x
    plt.subplot(3, 2, 6)
    plt.plot(t, (Jm_both_x), color='#00FF7F')
    plt.plot(t_array_cut, j_fit, '--', label=rf"$\tau = {tau:.2f}\ \mathrm{{fs}} \quad j_{{\rm final}} : j_{{\rm max}} = {ratio:.2f}$", color='r')
    plt.xlabel('Time [fs]')
    plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    plt.title('Δ matter current density along x')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()




    # # Суммарный ток по y
    # plt.subplot(5, 2, 9)
    # plt.plot(t, Jm_both_y, color='#00BFFF')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    # plt.title('Full matter current density along y')
    # plt.grid(True)
    # plt.tight_layout()


    # # Разностный ток по y
    # plt.subplot(5, 2, 10)
    # plt.plot(t, (Jm_both_y - Jm_pump_y - Jm_probe_y), color='#FF007F')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    # plt.title('Δ matter current density along y')
    # plt.grid(True)
    # plt.tight_layout()





    # lr_data = pd.read_csv(
    #     folder + '/lr_response.data',
    #     comment='#',
    #     delim_whitespace=True,
    #     header=None
    # )

    # lr_data.columns = [
    #     'Energy',        # 1
    #     'Re_sigma_x',    # 2
    #     'Re_sigma_y',    # 3
    #     'Re_sigma_z',    # 4
    #     'Im_sigma_x',    # 5
    #     'Im_sigma_y',    # 6
    #     'Im_sigma_z',    # 7
    #     'Re_eps_x',      # 8
    #     'Re_eps_y',      # 9
    #     'Re_eps_z',      # 10
    #     'Im_eps_x',      # 11
    #     'Im_eps_y',      # 12
    #     'Im_eps_z',      # 13
    # ]


    # # Построение графика изменения энергии во времени
    # plt.subplot(5, 2, 9)
    # plt.plot(lr_data['Energy'][100:], lr_data['Im_eps_z'][100:], label='Im(ε)', color='#00BFFF')
    # plt.xlabel('E [eV]')
    # plt.ylabel('Im(ε)')
    # plt.title('Absorption')
    # plt.grid(True)
    # #plt.legend()
    # plt.tight_layout()

    # # Построение графика изменения энергии во времени
    # plt.subplot(5, 2, 10)
    # plt.plot(lr_data['Energy'][100:], lr_data['Re_eps_z'][100:], label='Re(ε)', color='#FF007F')
    # plt.xlabel('E [eV]')
    # plt.ylabel('Re(ε)')
    # plt.title('Dispersion')
    # plt.grid(True)
    # #plt.legend()
    # plt.tight_layout()
    # # plt.show()

    # plt.savefig("ims/all/" + folder + ".pdf")
    plt.savefig("all_data/current_energy/ims/all/png/" + str(j)+ '_current_energy_all_plots_' + folder + ".png", dpi=200)
    plt.savefig("all_data/current_energy/ims/all/pdf/" + str(j)+ '_current_energy_all_plots_' + folder + ".pdf", bbox_inches="tight")

    dt = np.diff(t, prepend=t[0])

    arr = np.array(np.cumsum(-2.64*El_f_probe_x[m0_cut_3:m0_cut_1+1000]*dt[m0_cut_3:m0_cut_1+1000]))
    a_start = arr[m0_cut_2-m0_cut_3]
    a_fin = arr[-1]

    plt.show()
    plt.close()


    each_num=10

    current_energy_array=np.array([ t[::each_num], El_f_pump_z[::each_num], El_f_probe_x[::each_num],
                                    Jm_both_z[::each_num],
                                    Jm_both_y[::each_num],
                                    Jm_both_x[::each_num],
                                    -integral_both_z[::each_num],
                                    -integral_both_x[::each_num]
                                    ])

    columns = ["time",  "El_f_pump_z", "El_f_probe_x",
               "Jm_both_z",
               "Jm_both_y",
               "Jm_both_x",
               "En_both_z",
               "En_both_x"]


    df = pd.DataFrame(current_energy_array.T, columns=columns)


    df.to_csv("all_data/current_energy/data/" + str(j)+ "_current_energy_data_" + folder + ".csv", index=False)

    with open("all_data/current_energy/data/" + str(j)+ "_current_energy_data_" + folder + ".csv", "w") as f:
        f.write("# Units: Time [fs], Current Jm [1/fs*Angstrom^2], Electric field El_f[V/Angstrom], Energy En[eV]\n")
        df.to_csv(f, index=False)

    print("For " + folder + " current&energy data saving in csv format is completed")




    return tau


# In[ ]:


# In[ ]:





# ## exp fit drude funcs

# In[ ]:





# In[117]:


def drude_current_v1(dummy_t, j0, j_final, tau):

    # tau = 6.65
    dt = 0.002

    t_probe, t_pump = get_pulse_durations(folder)
    m0_cut_1 = int(500*t_probe)

    data_both = pd.read_csv(
        folder + '/both_pulses_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )


    data_both.columns = ['time_fs', 'Ac_ext_x', 'Ac_ext_y', 'Ac_ext_z', 'E_ext_x', 'E_ext_y', 'E_ext_z',         
        'Ac_tot_x', 'Ac_tot_y', 'Ac_tot_z', 'E_tot_x', 'E_tot_y', 'E_tot_z',         
        'Jm_x', 'Jm_y', 'Jm_z']

    t = data_both['time_fs']
    # El_f_probe_x = data_both['E_ext_x']

    # --- Задаём поле в дискретных точках ---
    t_array = np.array(t[m0_cut_1:])         # сетка времени
    # E_array = -np.array(El_f_probe_x[m0_cut_1:]) 


    # --- Решение через явный шаг Эйлера ---

    j_euler = np.zeros_like(t_array)
    j_euler[0] = j0

    for k in range(len(t_array)-1):
        j_euler[k+1] = j_euler[k] + dt * (-(j_euler[k]-j_final)/tau)


    return j_euler[::50]


# In[ ]:





# In[118]:


def drude_exp_fit_v2_pump(dummy_t, j0, j_final, tau):


    t_probe, t_pump = get_pulse_durations(folder)
    # m0_cut_4 = int(500*(t_probe+t_pump)/2) 

    # tau = 6.65
    dt = 0.002

    data_both = pd.read_csv(
        folder + '/both_pulses_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )


    data_both.columns = ['time_fs', 'Ac_ext_x', 'Ac_ext_y', 'Ac_ext_z', 'E_ext_x', 'E_ext_y', 'E_ext_z',         
        'Ac_tot_x', 'Ac_tot_y', 'Ac_tot_z', 'E_tot_x', 'E_tot_y', 'E_tot_z',         
        'Jm_x', 'Jm_y', 'Jm_z']

    t = data_both['time_fs']
    # El_f_probe_x = data_both['E_ext_x']

    # --- Задаём поле в дискретных точках ---
    t_array = np.array(t[m0_cut_4:])         # сетка времени
    # E_array = -np.array(El_f_probe_x[m0_cut_1:]) 


    # --- Решение через явный шаг Эйлера ---

    j_euler = np.zeros_like(t_array)
    j_euler[0] = j0

    for k in range(len(t_array)-1):
        j_euler[k+1] = j_euler[k] + dt * (-(j_euler[k]-j_final)/tau)


    return j_euler[::50]


# In[ ]:





# In[ ]:





# ## extract

# In[ ]:





# In[119]:


def extract_c_num(s):
    start = s.index('c') + 1
    end = s.index('_', start)
    return int(s[start:end])

def extract_s_num(s):
    start = s.index('_s') + 2
    end = s.index('_', start)
    return float(s[start:end])

def extract_k_num(s):
    start = s.index('_k') + 2
    end = s.index('_', start)
    return int(s[start:end])

# def extract_a_num(s):
#     start = s.index('_a') + 2
#     end = s.index('_', start)
#     return float(s[start:end])

def extract_r_num(s):
    start = s.index('_r') + 2
    end = s.index('_', start)
    return float(s[start:end])

def extract_i_num(s):
    start = s.index('_i') + 2
    end = s.index('_', start)
    return float(s[start:end])

def extract_p(s):
    match = re.search(r'p([^_]+)', s)
    if match:
        return float(match.group(1).replace("d", "e"))

def extract_amorphous_value(name):
    tokens = name.split("_")
    for i, token in enumerate(tokens):
        if token == "amorphous" and i + 1 < len(tokens):
            return float(tokens[i + 1])


# In[ ]:





# In[120]:


def get_n_excited_electrons(folder):

    output_probe = 'None'
    if os.path.isfile(os.path.join(folder, 'probe_pulse_nex.data')):
        data = np.loadtxt(folder+'/probe_pulse_nex.data', comments="#")
        time  = data[:, 0]
        nelec = data[:, 1]
        nhole = data[:, 2]

        if time[-1]>25:
            output_probe = nelec[-1]

    output_pump = 'None'
    if os.path.isfile(os.path.join(folder, 'probe_pulse_nex.data')):
        data = np.loadtxt(folder+'/pump_pulse_nex.data', comments="#")
        time  = data[:, 0]
        nelec = data[:, 1]
        nhole = data[:, 2]

        if time[-1]>25:
            output_pump = nelec[-1]


    output_both = 'None'
    if os.path.isfile(os.path.join(folder, 'probe_pulse_nex.data')):
        data = np.loadtxt(folder+'/both_pulses_nex.data', comments="#")
        time  = data[:, 0]
        nelec = data[:, 1]
        nhole = data[:, 2]

        if time[-1]>25:
            output_both = nelec[-1]



    gs_file = glob.glob(folder + '/Si_gs*')

    with open(gs_file[0], 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'al(1:3)' in line:
            nums = [float(x.replace('d0', '')) for x in line.split('=')[1].split(',')]
            factor = nums[0]*nums[1]*nums[2]

    return output_both/factor, output_pump/factor, output_probe/factor


# In[ ]:





# In[121]:


def get_pulse_durations(folder):

    pulses_file = glob.glob(folder + '/Si_rt_both*')

    with open(pulses_file[0], 'r') as file:
            lines = file.readlines()

    for line in lines:
        if 'tw1' in line:
            nums = [float(x.replace('d0', '')) for x in line.split('=')[1].split(',')]
            t_probe = nums[0]

    for line in lines:
        if 'tw2' in line:
            nums = [float(x.replace('d0', '')) for x in line.split('=')[1].split(',')]
            t_pump = nums[0]

    return t_probe, t_pump


# In[ ]:





# ## sorted folders

# In[ ]:





# In[107]:


ax = 'z'

folders = [name for name in os.listdir('.') if os.path.isdir(os.path.join('.', name))]

# folders = [f for f in folders if f.startswith('c')]
folders = [f for f in folders 
           if f.startswith('c') 
           and os.path.isfile(os.path.join(f, 'both_pulses_pulse.data'))]
           # and not os.path.isdir(os.path.join(f, 'restart'))]


# print(folders)


if "amorphous" in folders[0]:
    sorted_folders = sorted(folders, key=extract_amorphous_value, reverse=True)

else:
    sorted_folders = sorted(folders, key=lambda s: (extract_p(s), extract_c_num(s), extract_r_num(s), extract_s_num(s), extract_k_num(s), extract_i_num(s)))

print("Sorted folders are:")
for f in sorted_folders:
    print(f)
print(" ")

with open("all_data/sorted_folders.txt", "w") as f:
    for item in sorted_folders:
        f.write(item + "\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## all plots

# In[108]:


taus_exp_x = []

j=0
for folder in sorted_folders:
    j=j+1

    t_probe, t_pump = get_pulse_durations(folder)
    m0_cut_4 = int(500*(t_probe+t_pump)/2) # pump exp fit, at the end of the pump pulse

    tau = plot_nonlinear_response_v4(folder, ax, j)
    taus_exp_x.append(tau)
    print("For " + folder + " all plots for current&energy are completed")

print(" ")


# In[ ]:





# In[ ]:





# In[ ]:





# In[114]:


# df


# In[113]:


# df = pd.read_csv("all_data/current_energy/data/" + str(j)+ "_current_energy_data_" + folder + ".csv", comment="#")
# df

# plt.figure()
# plt.plot(df['time'], df['El_f_pump_z'])
# plt.plot(df['time'], df['El_f_probe_x'])

# plt.figure()
# # plt.plot(df['time'], df['Jm_both_z'])
# plt.plot(df['time'], df['Jm_both_y'])
# plt.plot(df['time'], df['Jm_both_x'])

# plt.figure()
# # plt.plot(df['time'], df['En_both_z'])
# plt.plot(df['time'], df['En_both_x'])


# In[ ]:





# In[ ]:




