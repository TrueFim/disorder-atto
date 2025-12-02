#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import matplotlib.image as mpimg
import os
from PIL import Image
from itertools import product

from scipy.signal import fftconvolve

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Полностью отключить все предупреждения
warnings.filterwarnings("ignore")

os.makedirs("ims", exist_ok=True)
# os.makedirs("ims/current", exist_ok=True)
os.makedirs("ims/gabor", exist_ok=True)

plt.rcParams['font.family'] = 'serif'
import matplotlib


# In[8]:


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

def extract_a_num(s):
    start = s.index('_a') + 2
    end = s.index('_', start)
    return float(s[start:end])

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
        return match.group(1)


# In[9]:


folders = [name for name in os.listdir('.') if os.path.isdir(os.path.join('.', name))]

# folders = [f for f in folders if f.startswith('c')]
folders = [f for f in folders 
           if f.startswith('c') 
           and os.path.isfile(os.path.join(f, 'probe_pulse_rt.data')) 
           and not os.path.isdir(os.path.join(f, 'restart'))]


# print(folders)


# Сортировка: сначала по c, потом по s, потом по k
sorted_folders = sorted(folders, key=lambda s: (extract_p(s), extract_c_num(s), extract_r_num(s), extract_s_num(s), extract_k_num(s), extract_a_num(s), extract_i_num(s)))


# In[10]:


# ----- Габорово окно -----
def gabor_transform(x, f, sigma, Fs):
    """
    x     : сигнал
    f     : частота анализа в Гц
    sigma : ширина окна (в СЕКУНДАХ!)
    Fs    : частота дискретизации
    """
    n = len(x)
    tt = np.arange(n) / Fs
    window = np.exp(-(tt - tt.mean())**2 / (2*sigma**2)) * np.exp(1j * 2*np.pi*f*tt)
    return fftconvolve(x, window, mode='same')



def gabor_plot(folder):



    # Загрузка данных, пропуская строки с комментариями (начинаются с #)
    data_both = pd.read_csv(
        folder + '/both_pulses_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    data_probe = pd.read_csv(
        folder + '/probe_pulse_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # Назначим читаемые имена колонкам
    data_both.columns = ['time_fs', 'Ac_ext_x', 'Ac_ext_y', 'Ac_ext_z', 'E_ext_x', 'E_ext_y', 'E_ext_z',         
        'Ac_tot_x', 'Ac_tot_y', 'Ac_tot_z', 'E_tot_x', 'E_tot_y', 'E_tot_z',         
        'Jm_x', 'Jm_y', 'Jm_z']

    data_probe.columns = ['time_fs', 'Ac_ext_x', 'Ac_ext_y', 'Ac_ext_z', 'E_ext_x', 'E_ext_y', 'E_ext_z',         
        'Ac_tot_x', 'Ac_tot_y', 'Ac_tot_z', 'E_tot_x', 'E_tot_y', 'E_tot_z',         
        'Jm_x', 'Jm_y', 'Jm_z']


    t = data_both['time_fs']

    Jm_both_x  =  data_both['Jm_x']
    Jm_probe_x =  data_probe['Jm_x']


    t = np.array(t)
    signal = Jm_both_x - Jm_probe_x
    # ----- Автоматически определяем частоту дискретизации -----
    Fs = 1 / (t[1] - t[0])


    # Построим графики
    plt.figure(figsize=(12, 10))

    # Общий заголовок для всей фигуры
    plt.suptitle('Gabor Representation ' + ' '.join(re.split('[/_]', folder)), fontsize=14, y=0.99)

    # ----- Настройки частот -----
    freqs = np.linspace(0, 0.3, 200)   # частоты анализа (Гц)
    sigma = 5  # ширина временного окна в секундах (например, 50 мс)
    # ----- Вычисляем спектр -----
    G = np.array([np.abs(gabor_transform(signal, f, sigma, Fs)) for f in freqs])

    # ----- Визуализация -----
    plt.subplot(2, 1, 1)
    plt.imshow(G, aspect='auto', origin='lower',
               extent=[t[0], t[-1], freqs[0], freqs[-1]], cmap = "seismic")
    plt.xlabel("Time (fs)")
    plt.ylabel("Frequency [PHz]")
    plt.title("σ = 5 fs")
    plt.colorbar()
    plt.tight_layout()


    # ----- Настройки частот -----
    freqs = np.linspace(0, 3, 200)   # частоты анализа (Гц)
    sigma = 0.2  # ширина временного окна в секундах (например, 50 мс)
    # ----- Вычисляем спектр -----
    G = np.array([np.abs(gabor_transform(signal, f, sigma, Fs)) for f in freqs])

    # ----- Визуализация -----
    plt.subplot(2, 1, 2)
    plt.imshow(G, aspect='auto', origin='lower',
               extent=[t[0], t[-1], freqs[0], freqs[-1]], cmap = "seismic")
    plt.xlabel("Time (fs)")
    plt.ylabel("Frequency [PHz]")
    plt.title("σ = 0.2 fs")
    plt.colorbar()
    plt.tight_layout()


    plt.savefig('ims/gabor/Gabor_' + folder + ".png", dpi=200)
    plt.show()


# In[11]:


for folder in sorted_folders:
    gabor_plot(folder)


# In[ ]:




