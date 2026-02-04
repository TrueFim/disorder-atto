#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import matplotlib.image as mpimg
import os
import csv
from glob import glob
from PIL import Image
from itertools import product

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from functools import partial

# Полностью отключить все предупреждения
warnings.filterwarnings("ignore")

os.makedirs("analysis", exist_ok=True)
# os.makedirs("ims/current", exist_ok=True)
os.makedirs("analysis/dos", exist_ok=True)

os.makedirs("analysis/dos/ims", exist_ok=True)
os.makedirs("analysis/dos/ims/full", exist_ok=True)
os.makedirs("analysis/dos/ims/bandgap", exist_ok=True)

os.makedirs("analysis/dos/data", exist_ok=True)
os.makedirs("analysis/dos/data", exist_ok=True)
os.makedirs("analysis/dos/data/full", exist_ok=True)
os.makedirs("analysis/dos/data/bandgap", exist_ok=True)


plt.rcParams['font.family'] = 'serif'
import matplotlib
# matplotlib.use('Agg')
# print(matplotlib.get_backend())

x_en_min = -7
x_en_max = 14.5

# число бинов до суммирования гауссов (ширина 0.01 эВ при int((x_en_max-x_en_min)*100))
n_bins = int((x_en_max-x_en_min)*100)

# во сколько раз ширина гаусса больше ширины бина (x_en_max-x_en_min)/n_bins
width_factor = 5

# число точек для графика после сглаживания (0.01 эВ между точками при int((x_en_max-x_en_min)*100))
num_points = int((x_en_max-x_en_min)*100)

# границы для графика вокруг bandgap
p1=0.5
p2=0.72


# In[2]:


with open("analysis/dos/params.txt", "w") as f:
    f.write("Energy boundaries: \n")
    f.write(f"E_min = {x_en_min}\n")
    f.write(f"E_max = {x_en_max}\n \n")
    f.write(f"Number of bins before Gaussian summation: \nn_bins = {n_bins}\n \n")
    f.write(f"Factor by which the Gaussian width exceeds the bin width: \nwidth_factor = {width_factor}\n \n")
    f.write(f"Number of points for the plot after smoothing: \nnum_points = {num_points}\n \n")
    f.write("Plot boundaries around the band gap: \n")
    f.write(f"p1 = {p1}\n")
    f.write(f"p2 = {p2}\n")


# In[ ]:





# ## extract

# In[3]:


def extract_first_symbol(s):
    return int(s[0])

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
    s1 = match.group(1).split('d')[0]
    s2 = match.group(1).split('d')[1]
    return float(s2+s1)
        # return match.group(1)[-2:]+match.group(1)[:-2]


# In[4]:


ax = 'z'

folders = [name for name in os.listdir('.') if os.path.isdir(os.path.join('.', name))]

# folders = [f for f in folders if f.startswith('c')]
# folders = [f for f in folders 
#            if f[0].isdigit() 
#            and os.path.isfile(os.path.join(f, 'probe_pulse_rt.data')) 
#            and not os.path.isdir(os.path.join(f, 'restart'))]


folders = [f for f in folders 
           if os.path.isfile(os.path.join(f, 'both_pulses_rt.data')) 
           and not os.path.isdir(os.path.join(f, 'restart'))]

# print(folders)


# Сортировка: сначала по c, потом по s, потом по k
sorted_folders = sorted(folders, key=lambda s: (extract_p(s), extract_c_num(s), extract_r_num(s), extract_s_num(s), extract_k_num(s), extract_a_num(s), extract_i_num(s)))

# sorted_folders = sorted(folders, key=lambda s: extract_first_symbol(s))


# In[5]:


# sorted_folders


# In[ ]:





# ## dos hist

# In[ ]:





# In[6]:


def get_numk_numstates(folder):

    gs_file = glob(folder + "/Si_gs*.inp")[0]

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





# In[7]:


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





# In[ ]:





# In[ ]:





# In[8]:


# folder = sorted_folders[0]
# filename = folder + '/gs_eigen.data'

# (gs_numk, gs_nstates) = get_numk_numstates(folder)


# bins_num = 200

# with open(filename, 'r') as f:
#     lines = f.readlines()

# startline_num=3

# for k_val in range(gs_numk):

#     energies = read_energies_for_k(lines, k_val+1, gs_nstates, startline_num)
#     startline_num = startline_num+gs_nstates+1


#     if (k_val+1)==gs_numk:

#         plt.figure(figsize=(8, 4))
#         plt.title(' '.join(re.split('[/_]', folder)), fontsize=14, y=1.03)
#         plt.hist(energies, bins=bins_num, color='b')  # bins можешь менять
#         plt.xlabel("1-particle energy (eV)")
#         plt.ylabel("Counts")
#         plt.ylim(0, 12)
#         plt.xlim(-7, 13.5)


# In[ ]:





# In[ ]:





# In[9]:


# for folder_name in sorted_folders:


#     filename = folder_name + '/gs_eigen.data'
#     (gs_numk, gs_nstates) = get_numk_numstates(folder_name)


#     n_bins = 1000
#     bins_vals = np.linspace(-7, 13.5, n_bins)
#     hist_sum = np.zeros(n_bins-1)


#     with open(filename, 'r') as f:
#         lines = f.readlines()

#     startline_num=3


#     for k_val in range(gs_numk):

#         energies = read_energies_for_k(lines, k_val+1, gs_nstates, startline_num)
#         startline_num = startline_num+gs_nstates+1
#         hist, _ = np.histogram(energies, bins=bins_vals)
#         hist_sum = hist_sum + hist

#         # if (k_val+1)%25==0:
#         if (k_val+1)==gs_numk:

#             plt.figure(figsize=(8, 4))
#             plt.title(' '.join(re.split('[/_]', folder_name)), fontsize=14, y=1.03)
#             plt.bar(bins_vals[:-1], 864*hist_sum/(k_val+1)/gs_nstates, width=np.diff(bins_vals), align='edge', color='b')
#             plt.xlabel("1-particle energy (eV)")
#             plt.ylabel("Counts")
#             plt.ylim(0, 15*200/n_bins)
#             plt.xlim(-7, 13.5)
#             plt.show()


# In[ ]:





# ## dos plots

# In[10]:


def delta_gauss(x, sigma, x0):
    return np.exp(-(x-x0)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)


# In[ ]:





# In[11]:


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

    plt.figure(figsize=(8, 4))

    if 'amorphous' not in folder_name:
        part_folder_name = folder_name.split('a', 1)[0].rstrip('_')
        plt.title(' '.join(re.split('[/_]', part_folder_name)), fontsize=14, y=1.03)
    else: plt.title(' '.join(re.split('[/_]', folder_name)), fontsize=14, y=1.03)

    plt.plot(energy_vals, delta_sum, color='b')
    plt.ylim(0, 4.5)
    plt.xlim(x_en_min, x_en_max)
    plt.grid(color='black', alpha=0.2)
    plt.xlabel("Energy [eV]")
    plt.ylabel("DOS [1/eV]")

    plt.savefig("analysis/dos/ims/full/" + folder_name + ".png", dpi=200)

    with open("analysis/dos/data/full/dos_full_" + folder_name + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Energy [eV]", "DOS [1/eV]"])   # заголовок (по желанию)
        writer.writerows(zip(energy_vals, delta_sum))




    energy_vals = np.linspace(x_en_min, x_en_max, num_points)
    del_width = (bins_centers[1]-bins_centers[0])

    delta_sum = np.zeros(num_points)

    for bin_center, hist_val in zip(bins_centers, hist_sum_cell):
        delta = delta_gauss(energy_vals, del_width/0.2, bin_center)

        delta_sum = delta_sum + hist_val*delta

    plt.figure(figsize=(8, 4))

    if 'amorphous' not in folder_name:
        part_folder_name = folder_name.split('a', 1)[0].rstrip('_')
        plt.title(' '.join(re.split('[/_]', part_folder_name)), fontsize=14, y=1.03)
    else: plt.title(' '.join(re.split('[/_]', folder_name)), fontsize=14, y=1.03)

    plt.plot(energy_vals[int(num_points*p1):int(num_points*p2)], delta_sum[int(num_points*p1):int(num_points*p2)], color='b')
    # plt.plot(energy_vals, delta_sum, color='b')
    plt.ylim(0, 4.5)
    plt.grid(color='black', alpha=0.2)
    plt.xlabel("Energy [eV]")
    plt.ylabel("DOS [1/eV]")

    plt.savefig("analysis/dos/ims/bandgap/" + folder_name + ".png", dpi=200)

    with open("analysis/dos/data/bandgap/dos_bandgap_" + folder_name + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Energy [eV]", "DOS [1/eV]"])   # заголовок (по желанию)
        writer.writerows(zip(energy_vals[int(num_points*p1):int(num_points*p2)], delta_sum[int(num_points*p1):int(num_points*p2)]))


# In[ ]:





# In[12]:


# folder_name=sorted_folders[3]
# folder_name

# x, y = np.loadtxt("analysis/dos/data/bandgap/" + folder_name + ".csv", delimiter=",", skiprows=1, unpack=True)

# plt.plot(x,y)


# In[ ]:





# In[ ]:





# In[ ]:




