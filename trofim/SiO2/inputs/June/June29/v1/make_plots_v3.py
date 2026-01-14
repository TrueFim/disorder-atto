#!/usr/bin/env python
# coding: utf-8

# ### Import

# In[115]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import matplotlib.image as mpimg
import os
from PIL import Image
from itertools import product

# Полностью отключить все предупреждения
warnings.filterwarnings("ignore")

os.makedirs("ims", exist_ok=True)
os.makedirs("ims/current", exist_ok=True)
os.makedirs("ims/all", exist_ok=True)



# In[116]:


plt.rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('Agg')
# print(matplotlib.get_backend())


# ### Nonlinear

# In[117]:


def plot_nonlinear_response_v1(folder, ax):

    # Загрузка данных, пропуская строки с комментариями (начинаются с #)
    data = pd.read_csv(
        folder + '/pulse_1e12_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # Назначим читаемые имена колонкам
    data.columns = [
        'time_fs',         # 1
        'Ac_ext_x',        # 2
        'Ac_ext_y',        # 3
        'Ac_ext_z',        # 4
        'E_ext_x',         # 5
        'E_ext_y',         # 6
        'E_ext_z',         # 7
        'Ac_tot_x',        # 8
        'Ac_tot_y',        # 9
        'Ac_tot_z',        #10
        'E_tot_x',         #11
        'E_tot_y',         #12
        'E_tot_z',         #13
        'Jm_x',         #14
        'Jm_y',         #15
        'Jm_z',         #16
    ]

    t = data['time_fs']
    Jm = data['Jm_'+ax]
    El_f = data['E_ext_z']

    # 1:Time[fs] 2:Ac_ext_x[fs*V/Angstrom] 3:Ac_ext_y[fs*V/Angstrom] 4:Ac_ext_z[fs*V/Angstrom]
    #5:E_ext_x[V/Angstrom] 6:E_ext_y[V/Angstrom] 7:E_ext_z[V/Angstrom] 8:Ac_tot_x[fs*V/Angstrom]
    # 9:Ac_tot_y[fs*V/Angstrom] 10:Ac_tot_z[fs*V/Angstrom] 11:E_tot_x[V/Angstrom]
    # 12:E_tot_y[V/Angstrom] 13:E_tot_z[V/Angstrom]  14:Jm_x[1/fs*Angstrom^2] 15:Jm_y[1/fs*Angstrom^2] 16:Jm_z[1/fs*Angstrom^2]


    # Построим графики
    # plt.figure(figsize=(10, 7.5))
    plt.figure(figsize=(10, 2.5))

    # Общий заголовок для всей фигуры
    plt.suptitle(' '.join(re.split('[/_]', folder)), fontsize=14)

    # # Первый график: внешнее поле по z
    # plt.subplot(3, 2, 1)
    # plt.plot(data['time_fs'], data['E_ext_z'], label='E_ext_z', color='blue')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('E [V/Å]')
    # plt.title('External electric field z')
    # plt.grid(True)
    # #plt.legend()

    # Второй график: дипольный момент по z
    # plt.subplot(3, 2, 4)
    plt.subplot(1, 2, 2)
    plt.plot(data['time_fs'], data['Jm_'+ax], label='Dipole Moment z', color='magenta')
    plt.xlabel('Time [fs]')
    plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    plt.title('Full matter current density')
    plt.grid(True)
    #plt.legend()

    plt.tight_layout()

    # Загрузка данных с указанием имен столбцов
    energy_data = pd.read_csv(
        folder + '/pulse_1e12_rt_energy.data',
        comment='#',
        delim_whitespace=True,
        header=None,
        names=['Time_fs', 'E_total_eV', 'Delta_E_eV']
    )

    # Построение графика изменения энергии во времени
    # plt.subplot(3, 2, 3)
    plt.subplot(1, 2, 1)
    plt.plot(energy_data['Time_fs'], energy_data['Delta_E_eV'], label='ΔE = E_total - E_initial', color='r')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [eV]')
    plt.title('Full excitation energy per unit cell')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()

    del_E = energy_data['Delta_E_eV']

    # Загрузка данных с указанием имен столбцов
    data = pd.read_csv(
        folder + '/pulse_1e12_pulse.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # Назначим читаемые имена колонкам
    data.columns = [
        'energy_eV',           # 1
        'Re_Jm_x',             # 2
        'Re_Jm_y',             # 3
        'Re_Jm_z',             # 4
        'Im_Jm_x',             # 5
        'Im_Jm_y',             # 6
        'Im_Jm_z',             # 7
        'Abs2_Jm_x',           # 8
        'Abs2_Jm_y',           # 9
        'Abs2_Jm_z',           #10
        'Re_E_ext_x',          #11
        'Re_E_ext_y',          #12
        'Re_E_ext_z',          #13
        'Im_E_ext_x',          #14
        'Im_E_ext_y',          #15
        'Im_E_ext_z',          #16
        'Abs2_E_ext_x',        #17
        'Abs2_E_ext_y',        #18
        'Abs2_E_ext_z',        #19
        'Re_E_tot_x',          #20
        'Re_E_tot_y',          #21
        'Re_E_tot_z',          #22
        'Im_E_tot_x',          #23
        'Im_E_tot_y',          #24
        'Im_E_tot_z',          #25
        'Abs2_E_tot_x',        #26
        'Abs2_E_tot_y',        #27
        'Abs2_E_tot_z',        #28
    ]

    Jw2 = data['Abs2_Jm_'+ax]
    energy = data['energy_eV']
    # # Построение графика для 1 и 10 колонки
    # plt.subplot(3, 2, 2)
    # plt.plot(data['energy_eV'], data['Abs2_Jm_'+ax], label='', color='c')
    # plt.xlabel('Energy [eV]')
    # plt.ylabel('|J($\omega)|^2 [Å^{-4}]$')
    # plt.yscale('log')
    # plt.title('Power spectrum')
    # plt.grid(True)
    # #plt.legend()
    # plt.tight_layout()

        # Загрузка данных, пропуская строки с комментариями (начинаются с #)
    data = pd.read_csv(
        folder + '/pulse_1e6_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # Назначим читаемые имена колонкам
    data.columns = [
        'time_fs',         # 1
        'Ac_ext_x',        # 2
        'Ac_ext_y',        # 3
        'Ac_ext_z',        # 4
        'E_ext_x',         # 5
        'E_ext_y',         # 6
        'E_ext_z',         # 7
        'Ac_tot_x',        # 8
        'Ac_tot_y',        # 9
        'Ac_tot_z',        #10
        'E_tot_x',         #11
        'E_tot_y',         #12
        'E_tot_z',         #13
        'Jm_x',         #14
        'Jm_y',         #15
        'Jm_z',         #16
    ]

    J_nonl = Jm - 10**3*data['Jm_'+ax]

    # plt.subplot(3, 2, 6)
    # plt.plot(data['time_fs'], Jm - 10**3*data['Jm_'+ax], label='', color='orange')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('J [$fs^{-1} \cdot Å^{-2}$]')
    # plt.title('Nonlinear matter current density')
    # plt.grid(True)


    # Загрузка данных с указанием имен столбцов
    energy_data = pd.read_csv(
        folder + '/pulse_1e6_rt_energy.data',
        comment='#',
        delim_whitespace=True,
        header=None,
        names=['Time_fs', 'E_total_eV', 'Delta_E_eV']
    )

    # # Построение графика изменения энергии во времени
    # plt.subplot(3, 2, 5)
    # plt.plot(energy_data['Time_fs'], del_E -10**6*energy_data['Delta_E_eV'], label='ΔE = E_total - E_initial', color='lime')
    # plt.xlabel('Time [fs]')
    # plt.ylabel('E [eV]')
    # plt.title('Nonlinear excitation energy per unit cell')
    # plt.grid(True)
    # #plt.legend()
    # plt.tight_layout()
    # plt.show()

    plt.savefig("ims/current/" + folder + ".png")

    return t, energy, Jw2, El_f, J_nonl, Jm


# In[118]:


def plot_nonlinear_response_v2(folder, ax):

    # Загрузка данных, пропуская строки с комментариями (начинаются с #)
    data = pd.read_csv(
        folder + '/pulse_1e12_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # Назначим читаемые имена колонкам
    data.columns = [
        'time_fs',         # 1
        'Ac_ext_x',        # 2
        'Ac_ext_y',        # 3
        'Ac_ext_z',        # 4
        'E_ext_x',         # 5
        'E_ext_y',         # 6
        'E_ext_z',         # 7
        'Ac_tot_x',        # 8
        'Ac_tot_y',        # 9
        'Ac_tot_z',        #10
        'E_tot_x',         #11
        'E_tot_y',         #12
        'E_tot_z',         #13
        'Jm_x',         #14
        'Jm_y',         #15
        'Jm_z',         #16
    ]

    t = data['time_fs']
    Jm = data['Jm_'+ax]
    El_f = data['E_ext_z']

    # 1:Time[fs] 2:Ac_ext_x[fs*V/Angstrom] 3:Ac_ext_y[fs*V/Angstrom] 4:Ac_ext_z[fs*V/Angstrom]
    #5:E_ext_x[V/Angstrom] 6:E_ext_y[V/Angstrom] 7:E_ext_z[V/Angstrom] 8:Ac_tot_x[fs*V/Angstrom]
    # 9:Ac_tot_y[fs*V/Angstrom] 10:Ac_tot_z[fs*V/Angstrom] 11:E_tot_x[V/Angstrom]
    # 12:E_tot_y[V/Angstrom] 13:E_tot_z[V/Angstrom]  14:Jm_x[1/fs*Angstrom^2] 15:Jm_y[1/fs*Angstrom^2] 16:Jm_z[1/fs*Angstrom^2]


    # Построим графики
    plt.figure(figsize=(10, 7.5))

    # Общий заголовок для всей фигуры
    plt.suptitle(' '.join(re.split('[/_]', folder)), fontsize=14)

    # Первый график: внешнее поле по z
    plt.subplot(3, 2, 1)
    plt.plot(data['time_fs'], data['E_ext_z'], label='E_ext_z', color='blue')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [V/Å]')
    plt.title('External electric field z')
    plt.grid(True)
    #plt.legend()

    # Второй график: дипольный момент по z
    plt.subplot(3, 2, 4)
    plt.plot(data['time_fs'], data['Jm_'+ax], label='Dipole Moment z', color='magenta')
    plt.xlabel('Time [fs]')
    plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    plt.title('Full matter current density')
    plt.grid(True)
    #plt.legend()

    plt.tight_layout()

    # Загрузка данных с указанием имен столбцов
    energy_data = pd.read_csv(
        folder + '/pulse_1e12_rt_energy.data',
        comment='#',
        delim_whitespace=True,
        header=None,
        names=['Time_fs', 'E_total_eV', 'Delta_E_eV']
    )

    # Построение графика изменения энергии во времени
    plt.subplot(3, 2, 3)
    plt.plot(energy_data['Time_fs'], energy_data['Delta_E_eV'], label='ΔE = E_total - E_initial', color='r')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [eV]')
    plt.title('Full excitation energy per unit cell')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()

    del_E = energy_data['Delta_E_eV']

    # Загрузка данных с указанием имен столбцов
    data = pd.read_csv(
        folder + '/pulse_1e12_pulse.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # Назначим читаемые имена колонкам
    data.columns = [
        'energy_eV',           # 1
        'Re_Jm_x',             # 2
        'Re_Jm_y',             # 3
        'Re_Jm_z',             # 4
        'Im_Jm_x',             # 5
        'Im_Jm_y',             # 6
        'Im_Jm_z',             # 7
        'Abs2_Jm_x',           # 8
        'Abs2_Jm_y',           # 9
        'Abs2_Jm_z',           #10
        'Re_E_ext_x',          #11
        'Re_E_ext_y',          #12
        'Re_E_ext_z',          #13
        'Im_E_ext_x',          #14
        'Im_E_ext_y',          #15
        'Im_E_ext_z',          #16
        'Abs2_E_ext_x',        #17
        'Abs2_E_ext_y',        #18
        'Abs2_E_ext_z',        #19
        'Re_E_tot_x',          #20
        'Re_E_tot_y',          #21
        'Re_E_tot_z',          #22
        'Im_E_tot_x',          #23
        'Im_E_tot_y',          #24
        'Im_E_tot_z',          #25
        'Abs2_E_tot_x',        #26
        'Abs2_E_tot_y',        #27
        'Abs2_E_tot_z',        #28
    ]

    Jw2 = data['Abs2_Jm_'+ax]
    energy = data['energy_eV']
    # Построение графика для 1 и 10 колонки
    plt.subplot(3, 2, 2)
    plt.plot(data['energy_eV'], data['Abs2_Jm_'+ax], label='', color='c')
    plt.xlabel('Energy [eV]')
    plt.ylabel('|J($\\omega)|^2 [Å^{-4}]$')
    plt.yscale('log')
    plt.title('Power spectrum')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()

        # Загрузка данных, пропуская строки с комментариями (начинаются с #)
    data = pd.read_csv(
        folder + '/pulse_1e6_rt.data',
        comment='#',
        delim_whitespace=True,
        header=None
    )

    # Назначим читаемые имена колонкам
    data.columns = [
        'time_fs',         # 1
        'Ac_ext_x',        # 2
        'Ac_ext_y',        # 3
        'Ac_ext_z',        # 4
        'E_ext_x',         # 5
        'E_ext_y',         # 6
        'E_ext_z',         # 7
        'Ac_tot_x',        # 8
        'Ac_tot_y',        # 9
        'Ac_tot_z',        #10
        'E_tot_x',         #11
        'E_tot_y',         #12
        'E_tot_z',         #13
        'Jm_x',         #14
        'Jm_y',         #15
        'Jm_z',         #16
    ]

    J_nonl = Jm - 10**3*data['Jm_'+ax]

    plt.subplot(3, 2, 6)
    plt.plot(data['time_fs'], Jm - 10**3*data['Jm_'+ax], label='', color='orange')
    plt.xlabel('Time [fs]')
    plt.ylabel('J [$fs^{-1} \\cdot Å^{-2}$]')
    plt.title('Nonlinear matter current density')
    plt.grid(True)


    # Загрузка данных с указанием имен столбцов
    energy_data = pd.read_csv(
        folder + '/pulse_1e6_rt_energy.data',
        comment='#',
        delim_whitespace=True,
        header=None,
        names=['Time_fs', 'E_total_eV', 'Delta_E_eV']
    )

    # Построение графика изменения энергии во времени
    plt.subplot(3, 2, 5)
    plt.plot(energy_data['Time_fs'], del_E -10**6*energy_data['Delta_E_eV'], label='ΔE = E_total - E_initial', color='lime')
    plt.xlabel('Time [fs]')
    plt.ylabel('E [eV]')
    plt.title('Nonlinear excitation energy per unit cell')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("ims/all/" + folder + ".png")

    return t, energy, Jw2, El_f, J_nonl, Jm


# In[ ]:





# ### Plots

# In[119]:


if __name__ == "__main__":

    ax = 'z'
    
    folders = [name for name in os.listdir('.') if os.path.isdir(os.path.join('.', name))]
    
    folders = [f for f in folders 
           if f.startswith('c') 
           and os.path.isfile(os.path.join(f, 'pulse_1e12_rt.data')) 
           and not os.path.isdir(os.path.join(f, 'restart'))]
    
    # print(folders)
    
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
    
    # Сортировка: сначала по c, потом по s, потом по k
    sorted_folders = sorted(folders, key=lambda s: (extract_c_num(s), extract_s_num(s), extract_r_num(s), extract_k_num(s), extract_a_num(s)))
    
    # sorted_folders
    
    
    # In[124]:
    
    
    for folder in sorted_folders:
        t, E, Jw2_am, El_f, J_nonl, J = plot_nonlinear_response_v1(folder, ax)
    
    for folder in sorted_folders:
        t, E, Jw2_am, El_f, J_nonl, J = plot_nonlinear_response_v2(folder, ax)
    
    
    # In[125]:
    
    
    ims_current = [name for name in os.listdir('ims/current') if name.lower().endswith('.png') and os.path.isfile(os.path.join('ims/current', name))]
    ims_current = sorted(ims_current, key=lambda s: (extract_c_num(s), extract_s_num(s), extract_k_num(s), extract_a_num(s)))
    
    ims_all = [name for name in os.listdir('ims/all') if name.lower().endswith('.png') and os.path.isfile(os.path.join('ims/all', name))]
    ims_all = sorted(ims_all, key=lambda s: (extract_c_num(s), extract_s_num(s), extract_k_num(s), extract_a_num(s)))
    
    
    # In[126]:
    
    
    def create_pdf(folder, ims):
    
        # Список PNG-картинок (в нужном порядке)
        png_files = [('ims/' + folder + '/' + im) for im in ims]
    
        # Загружаем изображения и конвертируем в RGB
        images = [Image.open(f).convert('RGB') for f in png_files]
    
        # Вычисляем общую высоту и максимальную ширину
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
    
        # Создаём новое изображение (белый фон)
        combined = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    
        # Вставляем изображения одно под другим
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
    
        # Сохраняем в PDF
        combined.save('ims/' + folder + '/combined_' + folder + '.pdf')
    
    
    # In[127]:
    
    
    create_pdf('current', ims_current)


# In[ ]:





# In[ ]:




