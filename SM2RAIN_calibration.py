import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob as g
from tqdm import tqdm
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cartopy.crs as ccrs
import scipy
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from datetime import timedelta

# xr.set_options(display_style="html")
xr.set_options(display_style="text")

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)
SM_cdict = {
    'red': [(0.0, 1.0, 1.0),
            (0.16666666666666666, 1.0, 1.0),
            (0.3333333333333333, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.6666666666666666, 0.0, 0.0),
            (0.8333333333333333, 0.0, 0.0),
            (1.0, 0.26666666666666666, 0.26666666666666666)],
    'green': [(0.0, 0.6352941176470588, 0.6352941176470588),
              (0.16666666666666666, 1.0, 1.0),
              (0.3333333333333333, 0.5098039215686274, 0.5098039215686274),
              (0.5, 1.0, 1.0),
              (0.6666666666666666, 0.0, 0.0),
              (0.8333333333333333, 0.0, 0.0),
              (1.0, 0.26666666666666666, 0.26666666666666666)],
    'blue': [(0.0, 0.0, 0.0),
             (0.16666666666666666, 0.0, 0.0),
             (0.3333333333333333, 0.0, 0.0),
             (0.5, 1.0, 1.0),
             (0.6666666666666666, 1.0, 1.0),
             (0.8333333333333333, 0.44313725490196076, 0.44313725490196076),
             (1.0, 0.3137254901960784, 0.3137254901960784)]
             }

cmap_SM = mpl.colors.LinearSegmentedColormap('Soil Moisture in Volumetric Unit [0, 0.6]',SM_cdict,256)


da_ASCAT = xr.open_dataset('/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/Korea/ASCAT_CDR_12p5km_Korea_260205.nc')
ASCAT_lat, ASCAT_lon = da_ASCAT.x.data, da_ASCAT.y.data

ASCAT_mean = da_ASCAT['ASCAT (Volumetric)'].sel(time = slice('2021','2023')).mean(dim = 'time')
ASCAT_std = da_ASCAT['ASCAT (Volumetric)'].sel(time = slice('2021','2023')).std(dim = 'time')

fig = plt.figure(figsize=(7,4),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = 0
gl.right_labels = 0
plt.pcolormesh(ASCAT_lon, ASCAT_lat, ASCAT_mean,cmap = cmap_SM, vmin =0,vmax = .6)
plt.colorbar()
plt.title('ASCAT SM Tavg [$m^3 / m^3$]')

def plot(SM, date, cmap = cmap_SM, vmax = 1):
    fig = plt.figure(figsize=(7,4),dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = 0
    gl.right_labels = 0
    plt.pcolormesh(ASCAT_lon, ASCAT_lat, SM,cmap = cmap, vmin =0,vmax = vmax)
    plt.colorbar()
    date_str = datetime.strftime(date, '%Y-%m-%d')
    plt.title(f'SM2Rain ({date_str}) [mm/day]')

def SM2Rain(SMt, SMtm1, Z, A, B):
    return Z*(SMt - SMtm1) + (a*SMt**b)

def get_SM2rain(date, Z, A ,B):    
    SMtm1 = da_ASCAT['ASCAT (Degree of Saturation)'].sel(time = date - timedelta(days=1))
    SMt = da_ASCAT['ASCAT (Degree of Saturation)'].sel(time = date)
    SM2RAIN_t = SM2Rain(SMtm1, SMt, Z, A, B)
    return SM2RAIN_t

A = 23
B = 44
Z = 87.5

date = pd.to_datetime('2021-05-26')
SM2RAIN_t = get_SM2rain(date, Z, A, B)

plot(SM2RAIN_t, date, cmap = 'jet', vmax = 10)
plot(P_IDW_2021['precipitation'].sel(time = date), date,cmap = 'jet', vmax = 10)

P_IDW_2021 = xr.open_dataset('/home/jaese/cpuserver_data/python_modules/kunhee/Results/SM2RAIN/Precipitation_IDW_2021.nc')
P_IDW_2022 = xr.open_dataset('/home/jaese/cpuserver_data/python_modules/kunhee/Results/SM2RAIN/Precipitation_IDW_2022.nc')
# P_IDW_2023 = xr.open_dataset('/home/jaese/cpuserver_data/python_modules/kunhee/Results/SM2RAIN/Precipitation_IDW_2023.nc')
# P_IDW_2024 = xr.open_dataset('/home/jaese/cpuserver_data/python_modules/kunhee/Results/SM2RAIN/Precipitation_IDW_2024.nc')

plt.imshow((P_IDW_2021['precipitation'].sel(time = '2021-04-21').data - SM2RAIN_210421.data),cmap = 'RdBu', vmin= -10, vmax = 10)


plt.hist(da_ASCAT['ASCAT (Degree of Saturation)'].sel(time = '2021').mean(dim='time').data.flatten(), alpha = .2, label = '2021')
plt.hist(da_ASCAT['ASCAT (Degree of Saturation)'].sel(time = '2022').mean(dim='time').data.flatten(), alpha = .2, label = '2022')
plt.legend()



base_FP = '/home/jaese'
cpuserver_data_FP = os.path.join(base_FP, 'cpuserver_data')
george_FP = os.path.join(cpuserver_data_FP, 'python_modules', 'kunhee')

# File paths
SM2RAIN_FP = os.path.join(george_FP, 'Results', 'SM2RAIN')
SM2RAIN_cal_FP = os.path.join(SM2RAIN_FP, 'SM2RAIN_cal.nc')
IDW_AWS_2021_FP = os.path.join(SM2RAIN_FP, 'Precipitation_IDW_AWS_2021.nc')
IDW_AWS_2022_FP = os.path.join(SM2RAIN_FP, 'Precipitation_IDW_AWS_2022.nc')
IDW_ASOS_2021_FP = os.path.join(SM2RAIN_FP, 'Precipitation_IDW_ASOS_2021.nc')
IDW_ASOS_2022_FP = os.path.join(SM2RAIN_FP, 'Precipitation_IDW_ASOS_2022.nc')
Thiessen_AWS_2021_FP = os.path.join(SM2RAIN_FP, 'Precipitation_Thiessen_AWS_2021.nc')
Thiessen_AWS_2022_FP = os.path.join(SM2RAIN_FP, 'Precipitation_Thiessen_AWS_2022.nc')
Thiessen_ASOS_2021_FP = os.path.join(SM2RAIN_FP, 'Precipitation_Thiessen_ASOS_2021.nc')
Thiessen_ASOS_2022_FP = os.path.join(SM2RAIN_FP, 'Precipitation_Thiessen_ASOS_2022.nc')
IDW_AWS_2021 = xr.open_dataset(IDW_AWS_2021_FP)
IDW_AWS_2022 = xr.open_dataset(IDW_AWS_2022_FP)
IDW_ASOS_2021 = xr.open_dataset(IDW_ASOS_2021_FP)
IDW_ASOS_2022 = xr.open_dataset(IDW_ASOS_2022_FP)
Thiessen_AWS_2021 = xr.open_dataset(Thiessen_AWS_2021_FP)
Thiessen_AWS_2022 = xr.open_dataset(Thiessen_AWS_2022_FP)
Thiessen_ASOS_2021 = xr.open_dataset(Thiessen_ASOS_2021_FP)
Thiessen_ASOS_2022 = xr.open_dataset(Thiessen_ASOS_2022_FP)
SM2RAIN_cal = xr.open_dataset(SM2RAIN_cal_FP)

import colormaps as cmaps
cmap_prec = cmaps.precip2_17lev

plot(IDW_AWS_2021['precipitation'].sum(dim = 'time'), pd.to_datetime('2021'), vmax = 3000, cmap = cmap_prec)

plt.imshow(IDW_ASOS_2021_FP)

lat = SM2RAIN_cal['latitude'].values
lon = SM2RAIN_cal['longitude'].values
SM2RAIN_cal_array = SM2RAIN_cal['SM2RAIN_array'].values

IDW_AWS_2021_array = IDW_AWS_2021['precipitation'].values
IDW_AWS_2022_array = IDW_AWS_2022['precipitation'].values
IDW_AWS_2021_2022_array = np.concatenate([IDW_AWS_2021_array, IDW_AWS_2022_array], axis=0)
IDW_AWS_array = np.transpose(IDW_AWS_2021_2022_array, (1, 2, 0))

IDW_ASOS_2021_array = IDW_ASOS_2021['precipitation'].values
IDW_ASOS_2022_array = IDW_ASOS_2022['precipitation'].values
IDW_ASOS_2021_2022_array = np.concatenate([IDW_ASOS_2021_array, IDW_ASOS_2022_array], axis=0)
IDW_ASOS_array = np.transpose(IDW_ASOS_2021_2022_array, (1, 2, 0))

Thiessen_AWS_2021_array = Thiessen_AWS_2021['precipitation'].values
Thiessen_AWS_2022_array = Thiessen_AWS_2022['precipitation'].values
Thiessen_AWS_2021_2022_array = np.concatenate([Thiessen_AWS_2021_array, Thiessen_AWS_2022_array], axis=0)
Thiessen_AWS_array = np.transpose(Thiessen_AWS_2021_2022_array, (1, 2, 0))

Thiessen_ASOS_2021_array = Thiessen_ASOS_2021['precipitation'].values
Thiessen_ASOS_2022_array = Thiessen_ASOS_2022['precipitation'].values
Thiessen_ASOS_2021_2022_array = np.concatenate([Thiessen_ASOS_2021_array, Thiessen_ASOS_2022_array], axis=0)
Thiessen_ASOS_array = np.transpose(Thiessen_ASOS_2021_2022_array, (1, 2, 0))

plt.imshow(IDW_AWS_diff[:,:,20], cmap = 'RdBu');plt.colorbar(); plt.clim(-20, 20)

IDW_AWS_diff = SM2RAIN_cal_array - IDW_AWS_array
IDW_ASOS_diff = SM2RAIN_cal_array - IDW_ASOS_array
Thiessen_AWS_diff = SM2RAIN_cal_array - Thiessen_AWS_array
Thiessen_ASOS_diff = SM2RAIN_cal_array - Thiessen_ASOS_array

IDW_AWS_diff_mean = np.nanmean(IDW_AWS_diff, axis=2)
IDW_ASOS_diff_mean = np.nanmean(IDW_ASOS_diff, axis=2)
Thiessen_AWS_diff_mean = np.nanmean(Thiessen_AWS_diff, axis=2)
Thiessen_ASOS_diff_mean = np.nanmean(Thiessen_ASOS_diff, axis=2)    

val_min = -4
val_max = 0

plt.imshow(IDW_AWS_diff_mean, cmap = 'RdBu', vmin = -10, vmax= 10);plt.colorbar()
plt.imshow(IDW_ASOS_diff_mean, cmap = 'RdBu', vmin = -10, vmax= 10);plt.colorbar()

plt.figure()
plt.hist(IDW_AWS_diff_mean.flatten(), label = 'SM2Rain (ASCAT) - AWS (AWS; IDW) [mm/day]', alpha = .5, bins= 100)
plt.hist(IDW_ASOS_diff_mean.flatten(), label = 'SM2Rain (ASCAT) - AWS (ASOS; IDW) [mm/day]', alpha = .5, bins= 100)
plt.axvline(np.nanmean(IDW_AWS_diff_mean), label = 'mean (SM2Rain (ASCAT) - AWS (AWS; IDW)) [mm/day]', c = 'tab:blue', linewidth = 3)
plt.axvline(np.nanmean(IDW_ASOS_diff_mean), label = 'mean (SM2Rain (ASCAT) - AWS (ASOS; IDW)) [mm/day]', c = 'tab:orange', linewidth = 3)
plt.axvline(0, c = 'k', linewidth = 3)
plt.grid(alpha = .5)
plt.xlim(-10, 10)
plt.legend()





fig_AWS, ax_AWS = hPlot.plot_map_Korea(
    lon, lat,
    IDW_AWS_diff_mean,
    val_min, val_max,
    plot_title='SM2RAIN 기반 강수량 예측 비교',
    label_title='SM2RAIN - 역거리 가중법 기반 강수량(AWS) (mm)',
)
plt.savefig(os.path.join(george_FP, 'Figures', 'SM2RAIN', 'SM2RAIN_comparison_AWS.png'))
plt.close(fig_AWS)

fig_ASOS, ax_ASOS = hPlot.plot_map_Korea(
    lon, lat,
    IDW_ASOS_diff_mean,
    val_min, val_max,
    plot_title='SM2RAIN 기반 강수량 예측 비교',
    label_title='SM2RAIN - 역거리 가중법 기반 강수량(ASOS) (mm)',
)
plt.savefig(os.path.join(george_FP, 'Figures', 'SM2RAIN', 'SM2RAIN_comparison_ASOS.png'))
plt.close(fig_ASOS)

fig_Thiessen_AWS, ax_Thiessen_AWS = hPlot.plot_map_Korea(
    lon, lat,
    Thiessen_AWS_diff_mean,
    val_min, val_max,
    plot_title='SM2RAIN 기반 강수량 예측 비교',
    label_title='SM2RAIN - 티센 보간법 기반 강수량(AWS) (mm)',
)
plt.savefig(os.path.join(george_FP, 'Figures', 'SM2RAIN', 'SM2RAIN_comparison_Thiessen_AWS.png'))
plt.close(fig_Thiessen_AWS)

fig_Thiessen_ASOS, ax_Thiessen_ASOS = hPlot.plot_map_Korea(
    lon, lat,
    Thiessen_ASOS_diff_mean,
    val_min, val_max,
    plot_title='SM2RAIN 기반 강수량 예측 비교',
    label_title='SM2RAIN - 티센 보간법 기반 강수량(ASOS) (mm)',
)
plt.savefig(os.path.join(george_FP, 'Figures', 'SM2RAIN', 'SM2RAIN_comparison_Thiessen_ASOS.png'))
plt.close(fig_Thiessen_ASOS)    