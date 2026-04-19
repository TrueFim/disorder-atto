#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


import numpy as np
import scipy
from scipy import signal
from scipy.signal import hilbert
from scipy import integrate

from glob import glob
from PIL import Image
from itertools import product

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

os.makedirs("ims", exist_ok=True)
# os.makedirs("ims/current", exist_ok=True)
os.makedirs("ims/all", exist_ok=True)

# VERY CAREFULLY not to delete other folders!
if os.path.isdir("all_data"):
    shutil.rmtree("all_data")
    print("The all_data folder is deleted and created again")
    print(" ")

os.makedirs("dos_n_ex_data", exist_ok=True)

os.makedirs("dos_n_ex_data/dos", exist_ok=True)
os.makedirs("dos_n_ex_data/dos/data", exist_ok=True)
os.makedirs("dos_n_ex_data/dos/ims", exist_ok=True)
os.makedirs("dos_n_ex_data/dos/ims/png", exist_ok=True)
os.makedirs("dos_n_ex_data/dos/ims/pdf", exist_ok=True)

os.makedirs("dos_n_ex_data/n_ex", exist_ok=True)
os.makedirs("dos_n_ex_data/n_ex/data", exist_ok=True)

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





# ## extract funcs

# In[ ]:





# In[2]:


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





# In[3]:


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





# ## sorted folders

# In[ ]:





# In[4]:


ax = 'z'

folders = [name for name in os.listdir('.') if os.path.isdir(os.path.join('.', name))]

# folders = [f for f in folders if f.startswith('c')]
folders = [f for f in folders 
           if f.startswith('c') 
           and os.path.isfile(os.path.join(f, 'probe_pulse_rt.data'))]
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


# In[ ]:





# ## get n_ex

# In[ ]:





# In[5]:


# for f in sorted_folders:
#     print(get_n_excited_electrons(f))


# In[ ]:





# In[ ]:





# In[6]:


n_ex_data = []

for f in sorted_folders:
    n1, n2, n3 = get_n_excited_electrons(f)
    n_ex_data.append([f, n1, n2, n3])

# создаём DataFrame
df = pd.DataFrame(n_ex_data, columns=["folder", "both_n_ex", "pump_n_ex", "probe_n_ex"])

# делаем название папки индексом
df = df.set_index("folder")

# сохраняем в csv
df.to_csv("dos_n_ex_data/n_ex/data/n_ex_data.csv")


with open("dos_n_ex_data/n_ex/data/n_ex_data.csv", "w") as f:
    f.write("# Concentration of excited electrons in the conduction band [1/Å^3] \n")
    df.to_csv(f)

print("n_ex data is saved")


# In[ ]:





# In[ ]:





# ## dos

# In[ ]:





# In[7]:


x_en_min = -8
x_en_max = 15

# число бинов до суммирования гауссов (ширина 0.01 эВ при int((x_en_max-x_en_min)*100))
n_bins = int((x_en_max-x_en_min)*100)

# во сколько раз ширина гаусса больше ширины бина (x_en_max-x_en_min)/n_bins
width_factor = 5

# число точек для графика после сглаживания (0.01 эВ между точками при int((x_en_max-x_en_min)*100))
num_points = int((x_en_max-x_en_min)*100)

# границы для графика вокруг bandgap
p1=0.5
p2=0.72


# In[ ]:





# In[8]:


def get_numk_numstates(folder):

    gs_file = glob.glob(folder + "/Si_gs*.inp")[0]

    kprod = None
    nstate = None

    with open(gs_file) as f:
        for line in f:
            line = line.strip()

            if "num_kgrid" in line:
                # ex. num_kgrid(1:3) = 8, 16, 16
                nums = line.split("=")[1].split(",")
                kx, ky, kz = map(int, nums)
                kprod = kx * ky * kz

            if "nstate" in line:
                # ex. nstate = 64
                nstate = int(line.split("=")[1])

    # print("k-point product:", kprod)
    # print("nstate:", nstate)

    return (kprod, nstate)


# In[ ]:





# In[ ]:





# In[9]:


def read_energies_for_k(lines, k_target, n_levels, startline_num):
    """
    Возвращает массив энергий (eV) длины n_levels для заданного k
    """
    energies = []
    reading = False

    idx=0
    for line in lines[startline_num:startline_num+n_levels+1]:
        line = line.strip()

        # начало блока
        if (not reading) and line.startswith('k='):
            k_current = int(line.split(',')[0].split('=')[1])
            reading = (k_current == k_target)
            continue

        # читаем строки с энергиями
        if reading:
            if len(energies) < n_levels:
                parts = line.split()
                energy = float(parts[1])  # столбец esp[eV]
                energies.append(energy)
            else:
                break

    if len(energies) != n_levels:
        raise ValueError(f'Для k={k_target} найдено {len(energies)} уровней, ожидалось {n_levels}')

    return np.array(energies)


# In[ ]:





# In[10]:


def delta_gauss(x, sigma, x0):
    return np.exp(-(x-x0)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)


# In[ ]:





# In[11]:


def blue_red_gradient(n):
    """
    Возвращает список из n цветов (hex),
    создающих линейный градиент от синего к красному.
    """
    colors = []

    for i in range(n):
        t = i / (n - 1)  # параметр от 0 до 1

        r = t            # красный растёт
        g = 0            # зелёный фиксирован
        b = 1 - t        # синий убывает

        colors.append(to_hex((r, g, b)))

    return colors


# In[ ]:





# In[12]:


plt.figure(figsize=(8, 4))

colors = blue_red_gradient(len(sorted_folders))
j=0

for folder_name in sorted_folders:

    filename = folder_name + '/gs_eigen.data'
    (gs_numk, gs_nstates) = get_numk_numstates(folder_name)


    bins_vals = np.linspace(x_en_min, x_en_max, n_bins)
    hist_sum = np.zeros(n_bins-1)


    with open(filename, 'r') as f:
        lines = f.readlines()

    startline_num=3


    for k_val in range(gs_numk):

        energies = read_energies_for_k(lines, k_val+1, gs_nstates, startline_num)
        startline_num = startline_num+gs_nstates+1
        hist, _ = np.histogram(energies, bins=bins_vals)
        hist_sum = hist_sum + hist


        # if (k_val+1)==gs_numk:

        #     plt.figure(figsize=(8, 4))

        #     if 'amorphous' not in folder_name:
        #         part_folder_name = folder_name.split('a', 1)[0].rstrip('_')
        #         plt.title(' '.join(re.split('[/_]', part_folder_name)), fontsize=14, y=1.03)
        #     else: plt.title(' '.join(re.split('[/_]', folder_name)), fontsize=14, y=1.03)

        #     plt.bar(bins_vals[:-1], 864*hist_sum/(k_val+1)/gs_nstates, width=np.diff(bins_vals), align='edge', color='b')
        #     plt.xlabel("1-particle energy (eV)")
        #     plt.ylabel("Counts")
        #     plt.ylim(0, 15*200/n_bins)
        #     plt.xlim(x_en_min, x_en_max)
        #     plt.show()


    hist_sum_cell = hist_sum/gs_numk/(gs_nstates/32)
    bins_centers = (bins_vals[:-1]+bins_vals[1:])/2
    # print("Number of states :", np.sum(hist_sum_cell))


    energy_vals = np.linspace(x_en_min, x_en_max, num_points)
    del_width = (bins_centers[1]-bins_centers[0])

    delta_sum = np.zeros(num_points)

    for bin_center, hist_val in zip(bins_centers, hist_sum_cell):
        delta = delta_gauss(energy_vals, width_factor*del_width, bin_center)

        delta_sum = delta_sum + hist_val*delta


    # if 'amorphous' not in folder_name:
    #     part_folder_name = folder_name.split('a', 1)[0].rstrip('_')
    #     plt.title(' '.join(re.split('[/_]', part_folder_name)), fontsize=14, y=1.03)
    # else: plt.title(' '.join(re.split('[/_]', folder_name)), fontsize=14, y=1.03)

    plt.title("DOS comparison", fontsize=14, y=1.03)

    plt.plot(energy_vals, delta_sum, color=colors[j], label=folder_name)
    j=j+1
    plt.ylim(0, 4.5)
    plt.xlim(x_en_min, x_en_max)
    plt.grid(color='black', alpha=0.2)
    plt.xlabel("Energy [eV]")
    plt.ylabel("DOS [1/eV]")
    plt.legend(fontsize=8)


    plt.savefig("dos_n_ex_data/dos/ims/png/DOS_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig("dos_n_ex_data/dos/ims/pdf/DOS_comparison.pdf", bbox_inches="tight")



    dos_data=np.array([energy_vals, delta_sum])

    columns = ["energy", "dos"]
    df = pd.DataFrame(dos_data.T, columns=columns)


    df.to_csv("dos_n_ex_data/dos/data/" + str(j) + "_dos_data_" + folder_name + ".csv", index=False)

    with open("dos_n_ex_data/dos/data/" + str(j) + "_dos_data_" + folder_name + ".csv", "w") as f:
        f.write(r"# Units: Energy [eV], DOS [1/eV]" + "\n")
        df.to_csv(f, index=False)


    # energy_vals = np.linspace(x_en_min, x_en_max, num_points)
    # del_width = (bins_centers[1]-bins_centers[0])

    # delta_sum = np.zeros(num_points)

    # for bin_center, hist_val in zip(bins_centers, hist_sum_cell):
    #     delta = delta_gauss(energy_vals, del_width/0.2, bin_center)

    #     delta_sum = delta_sum + hist_val*delta

    # plt.figure(figsize=(8, 4))

    # if 'amorphous' not in folder_name:
    #     part_folder_name = folder_name.split('a', 1)[0].rstrip('_')
    #     plt.title(' '.join(re.split('[/_]', part_folder_name)), fontsize=14, y=1.03)
    # else: plt.title(' '.join(re.split('[/_]', folder_name)), fontsize=14, y=1.03)

    # plt.plot(energy_vals[int(num_points*p1):int(num_points*p2)], delta_sum[int(num_points*p1):int(num_points*p2)], color='b')
    # # plt.plot(energy_vals, delta_sum, color='b')
    # plt.ylim(0, 4.5)
    # plt.grid(color='black', alpha=0.2)
    # plt.xlabel("Energy [eV]")
    # plt.ylabel("DOS [1/eV]")

    # plt.savefig("analysis/dos/ims/bandgap/" + folder_name + ".png", dpi=200)

    # with open("analysis/dos/data/bandgap/" + folder_name + ".csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Energy [eV]", "DOS [1/eV]"])   # заголовок (по желанию)
    #     writer.writerows(zip(energy_vals[int(num_points*p1):int(num_points*p2)], delta_sum[int(num_points*p1):int(num_points*p2)]))

print("dos data is saved")


# In[ ]:





# In[ ]:




