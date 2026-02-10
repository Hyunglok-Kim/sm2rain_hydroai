import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import glob as g
from tqdm import tqdm
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cartopy.crs as ccrs
import scipy
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

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

# # ==============================================================================
# # preparation
# # ==============================================================================
ascat_info = xr.open_dataset('/home/jaese/NAS/ASCAT/TUW/warp5_grid/TUW_WARP5_grid_info_2_3.nc')
porosity = xr.open_dataset('/home/jaese/NAS/ASCAT/TUW/static_layer/porosity.nc')

lon_min, lon_max = 125, 129
lat_min, lat_max =  34,  39

margin = 1.0  # 도 단위 margin (원하면 0.5, 2 등으로 바꿔)

lon_min_m = lon_min - margin
lon_max_m = lon_max + margin
lat_min_m = lat_min - margin
lat_max_m = lat_max + margin

df_ascat_info = ascat_info.to_dataframe()

df_ascat_info = df_ascat_info[(df_ascat_info['lon'] > lon_min_m) & (df_ascat_info['lon'] < lon_max_m)]
df_ascat_info = df_ascat_info[(df_ascat_info['lat'] > lat_min_m) & (df_ascat_info['lat'] < lat_max_m)]

gids = df_ascat_info['cell'].unique().astype(int)

df_porosity = porosity.to_dataframe()

df_porosity = df_porosity[(df_porosity['lon'] > lon_min_m) & (df_porosity['lon'] < lon_max_m)]
df_porosity = df_porosity[(df_porosity['lat'] > lat_min_m) & (df_porosity['lat'] < lat_max_m)]

df_ascat_info['porosity_HWSD'] = df_porosity['por_hwsd']
df_ascat_info['porosity_GLDAS'] = df_porosity['por_gldas']

df_ascat_info['gpi'] = df_ascat_info['gpi'].astype(int)
gids = df_ascat_info['cell'].unique().astype(int)

df_ascat_info.index.name ='location_id'


fig = plt.figure(figsize=(7,4),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
gl = ax.gridlines(draw_labels=True)
gl.bottom_labels = 0
gl.right_labels = 0
plt.scatter(
    df_ascat_info['lon'],
    df_ascat_info['lat'],
    c = df_ascat_info['gpi'],
)

lon_min = df_ascat_info['lon'].min()
lon_max = df_ascat_info['lon'].max()
lat_min = df_ascat_info['lat'].min()
lat_max = df_ascat_info['lat'].max()

ASCAT_lon_ = np.arange(lon_min, lon_max, 0.125)
ASCAT_lat_ = np.arange(lat_min, lat_max, 0.125)

ASCAT_lon, ASCAT_lat = np.meshgrid(ASCAT_lon_, ASCAT_lat_)
ASCAT_lat = np.flipud(ASCAT_lat)
# plt.imshow(ASCAT_lon)
# plt.imshow(ASCAT_lat)

# # SMAP grid
# # da_vars_mean = xr.open_dataset('/home/jaese/cpuserver_data/personal_data/jaese/PINN/data/da_vars_mean.nc')

# # Lon = da_vars_mean.lon.data
# # Lat = da_vars_mean.lat.data
# # Lat_conus = Lat[41:116, 147:301]
# # Lon_conus = Lon[41:116, 147:301]
# # ==============================================================================
# # get stacked ASCAT 
# # ==============================================================================
# gids = [2220, 2221]
# ASCATlist = [g.glob(f'/home/jaese/NAS/ASCAT/TUW/csv/h119_h120_{x}.csv') for x in gids] 
# # ASCAT_2220 = pd.read_csv'/home/jaese/NAS/ASCAT/TUW/csv/h119_h120_2220.csv'
# ASCATlist = [x[0] for x in ASCATlist if x]
# dflist = []

# for i in tqdm(ASCATlist):
#     df = pd.read_csv(i, index_col = 0)
#     dflist.append(df)
# df.head()

# ASCAT_all = pd.concat(df_list, ignore_index=True)
# ASCAT_all.to_csv('/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/ASCAT_Korea.csv')

ASCAT_all = pd.read_csv('/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/ASCAT_Korea.csv', index_col = 0)

# ==============================================================================
# get ASCAT daily image
# ==============================================================================
time = pd.to_datetime(
        ASCAT_all['local_time'],
        unit='D',
        origin='1900-01-01',
        # utc=True
    )
time_date = time.dt.floor('D')

# time2 = pd.to_datetime(
#         ASCAT_all['time'],
#         unit='D',
#         origin='1900-01-01',
#         # utc=True
#     )
# time_date2 = time2.dt.floor('D')
# time.to_csv("/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/ascat_time.csv")

# time_date_cat = time_date.astype('category')
# time_date_cat.to_csv("/home/jaese/cpuserver_data/personal_data/jaese/ASCAT2SMAP/data/ASCAT-HSAF/ascat_time_cat.csv")
# ==============================================================================
# Resample
# ==============================================================================
def ascat_to_grid_nearest_thresh(df_merged,
                                 value_col,
                                 grid_lon,
                                 grid_lat,
                                 res_deg=0.125,
                                 factor=1.5):
    """
    grid 격자점 기준으로 최근접 ASCAT 값을 가져오되,
    거리가 res_deg * factor 를 초과하면 NaN 으로 남겨두는 함수.

    df_merged : lon, lat, value_col 컬럼을 가진 DataFrame
    value_col: 'SM_vol' 등
    grid_lon, grid_lat : 2D 격자 (예: ASCAT_lon, ASCAT_lat)
    res_deg  : 격자 해상도 (deg) 0.25, 0.125 등
    factor   : 허용 배수 (1.5 → 해상도의 1.5배까지 허용)
    """

    # ASCAT 포인트 좌표 & 값
    pts_asc = np.column_stack([df_merged['lon'].values,
                               df_merged['lat'].values])
    vals = df_merged[value_col].values

    # NaN 값은 미리 제거
    valid = np.isfinite(vals)
    pts_asc = pts_asc[valid]
    vals    = vals[valid]

    ny, nx = grid_lon.shape
    out = np.full((ny, nx), np.nan, dtype='float32')

    if len(vals) == 0:
        # 유효한 ASCAT 값이 하나도 없으면 그냥 NaN 필드 반환
        return out

    # 격자점 좌표 (질문자)
    pts_grid = np.column_stack([grid_lon.ravel(),
                                grid_lat.ravel()])

    tree = cKDTree(pts_asc)
    dist, idx = tree.query(pts_grid, k=1)   # dist: (Ngrid,)

    max_dist = res_deg * factor

    # threshold 안쪽에 있는 격자만 값 채우기
    ok = dist <= max_dist

    flat_out = out.ravel()
    flat_out[ok] = vals[idx[ok]].astype('float32')

    return flat_out.reshape(ny, nx)


outpath_Vol = '/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/Korea/Vol/'
outpath_Sat = '/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/Korea/Sat/'
outpath_VV = '/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/Korea/VV/'
# os.makedirs(outpath_Vol)
# os.makedirs(outpath_Sat)
# os.makedirs(outpath_VV)

datelist = sorted(time_date.unique())
i = 2000
for i in tqdm(range(len(datelist))):
    
    outdate = pd.Timestamp(datelist[i]).strftime('%Y%m%d')
    ascat_t = ASCAT_all[time_date == pd.Timestamp(datelist[i])]
    ascat_t.dropna(inplace=True)
    ascat_t_mean = ascat_t.groupby('location_id').mean()
    
    df_merged = df_ascat_info.join(
    ascat_t_mean['sigma40'],
    how='left',
    )
    # df_merged = df_ascat_info.join(
    # ascat_t_mean['sm'],
    # how='left',
    # )
    # df_merged['SM_vol'] = (df_merged['sm'].values/ 100) * df_merged['porosity_HWSD'].values
    # df_merged['SM_sat'] = (df_merged['sm'].values/ 100)# * df_merged['porosity_HWSD'].values
    df_merged['VV'] = df_merged['sigma40'].values# * df_merged['porosity_HWSD'].values
    # df_merged.dropna()
    
    # ASCAT_12p5km_Korea_vol = ascat_to_grid_nearest_thresh(df_merged, value_col='SM_vol',grid_lon=ASCAT_lon,grid_lat=ASCAT_lat,res_deg =0.125, factor=1.5)
    # ASCAT_12p5km_Korea_sat = ascat_to_grid_nearest_thresh(df_merged, value_col='SM_sat',grid_lon=ASCAT_lon,grid_lat=ASCAT_lat,res_deg =0.125, factor=1.5)
    ASCAT_12p5km_Korea_VV = ascat_to_grid_nearest_thresh(df_merged, value_col='VV',grid_lon=ASCAT_lon,grid_lat=ASCAT_lat,res_deg =0.125, factor=1.5)
    # np.save(outpath_Vol + f'ASCAT_HSAF_12p5km_Korea_Volumetric_{outdate}.npy', ASCAT_12p5km_Korea_vol)
    # np.save(outpath_Sat + f'ASCAT_HSAF_12p5km_Korea_Degree_Of_Saturation_{outdate}.npy', ASCAT_12p5km_Korea_sat)
    np.save(outpath_VV + f'ASCAT_HSAF_12p5km_Korea_VV_{outdate}.npy', ASCAT_12p5km_Korea_VV)


fig = plt.figure(figsize=(7,4),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = 0
gl.right_labels = 0
plt.pcolormesh(ASCAT_lon, ASCAT_lat, ASCAT_12p5km_Korea_VV, vmin =-30,vmax = 0)
plt.colorbar()
plt.title('ASCAT VV [dB]')


ASCAT_V = sorted(g.glob(outpath_Vol + f'*Volumetric*.npy'))
ASCAT_S = sorted(g.glob(outpath_Sat + f'*Degree_Of_Saturation*.npy'))
ASCAT_VV = sorted(g.glob(outpath_VV + f'*VV*.npy'))

ASCAT_V_arr = np.full((len(datelist), 55, 48), np.nan)
ASCAT_S_arr = np.full((len(datelist), 55, 48), np.nan)
ASCAT_VV_arr = np.full((len(datelist), 55, 48), np.nan)

for i in tqdm(range(len(datelist))):
    ASCAT_VV_arr_ = np.load(ASCAT_VV[i])
    ASCAT_V_arr_ = np.load(ASCAT_V[i])
    ASCAT_S_arr_ = np.load(ASCAT_S[i])
    ASCAT_VV_arr[i] = ASCAT_VV_arr_
    ASCAT_V_arr[i] = ASCAT_V_arr_
    ASCAT_S_arr[i] = ASCAT_S_arr_
    # plt.imshow(ASCAT_)
# plt.plot(ASCAT_arr[:, 40,40])

da_ASCAT = xr.Dataset(
        {
        'ASCAT (Volumetric)': (['time', 'lat', 'lon'], ASCAT_V_arr),
        'ASCAT (Degree of Saturation)': (['time', 'lat', 'lon'], ASCAT_S_arr),
        'ASCAT (VV)': (['time', 'lat', 'lon'], ASCAT_VV_arr),
        },
        coords={
            'time': pd.to_datetime(datelist),
            "y": (['lat', 'lon'], ASCAT_lat),
            "x": (['lat', 'lon'], ASCAT_lon),
            },  
        attrs=dict(
            description="ASCAT-CDR SM for South Korea (H119-H120)",
            units="Volumetric and Degree of Saturation SM",
        ),  
        ) 

da_ASCAT.to_netcdf('/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/Korea/ASCAT_CDR_12p5km_Korea_260209.nc')
# da_ASCAT.close()

# ==============================================================================
# Produce images
# ==============================================================================
da_ASCAT = xr.open_dataset('/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT/Korea/ASCAT_CDR_12p5km_Korea_260205.nc')
# ASCAT_lon_, ASCAT_lat_ = da_ASCAT.lon.data, da_ASCAT.lat.data
# ASCAT_lon, ASCAT_lat = np.meshgrid(ASCAT_lon_, ASCAT_lat_)
# ASCAT_lat = np.flipud(ASCAT_lat)

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

fig = plt.figure(figsize=(7,4),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = 0
gl.right_labels = 0
plt.pcolormesh(ASCAT_lon, ASCAT_lat, ASCAT_std,cmap = cmap_SM, vmin =0,vmax = .3)
plt.colorbar()
plt.title('ASCAT SM Tstd [$m^3 / m^3$]')

landmask = ~np.isnan(ASCAT_mean)
# plt.imshow(landmask)

four_river_basin = np.full((ASCAT_std.shape), np.nan)

four_river_basin[(ASCAT_lat<38.5) & (ASCAT_lat>37) & (ASCAT_lon>126)] = 1 # han-gang river basin
four_river_basin[(ASCAT_lat<37) & (ASCAT_lat>35.7) & (ASCAT_lon<128)] = 2 # gum gang river basin
four_river_basin[(ASCAT_lat<37) & (ASCAT_lon>128)] = 3 # nakdong gang river basin
four_river_basin[(ASCAT_lat<35.7) & (ASCAT_lon<128) & (ASCAT_lat<38.5)] = 4 # yougnsan gnang river basin
four_river_basin = np.where(landmask == 1, four_river_basin, np.nan)

# da_ASCAT.where(four_river_basin==1)


# fig = plt.figure(figsize=(7,4),dpi=150)
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines(resolution='10m')
# gl = ax.gridlines(draw_labels=True)
# gl.top_labels = 0
# gl.right_labels = 0
# # plt.pcolormesh(ASCAT_lon, ASCAT_lat, ASCAT_mean.where(four_river_basin==1),cmap = cmap_SM, vmin =0,vmax = .6)
# # plt.pcolormesh(ASCAT_lon, ASCAT_lat, landmask,cmap = "Blues", vmin =0,vmax = 4)
# im0 = plt.pcolormesh(ASCAT_lon, ASCAT_lat, four_river_basin,cmap = 'jet', vmin =1,vmax = 4)
# # plt.colorbar()
# plt.title('4 River basins')


from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

# 4개 클래스용 색(원하는 색으로 바꿔도 됨)
cmap4 = ListedColormap([
    "#d62728",  # Han
    "#ff7f0e",  # Geum
    # "#FFFF00",  # Nakdong
    "#FFE135",  # Nakdong
    "#008000",  # Yeongsan
])
# 값이 1,2,3,4 라고 가정 -> 경계는 0.5 간격
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap4.N)


fig = plt.figure(figsize=(7,4),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = 0
gl.right_labels = 0

im0 = ax.pcolormesh(
    ASCAT_lon, ASCAT_lat, four_river_basin,
    cmap=cmap4, norm=norm
)

cbar = fig.colorbar(im0, ax=ax, ticks=[1,2,3,4], orientation='vertical')
river_names = ['Han', 'Geum', 'Nakdong', 'Yeongsan']
cbar.ax.set_yticklabels(river_names)

plt.title('4 River basins')
plt.show()

da_ASCAT_ = da_ASCAT.sel(time = slice('2021', '2023'))

fig,axs = plt.subplots(4,1, figsize= (8,8))
axs[0].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==1)['ASCAT (Volumetric)'].mean(dim = ('lat', 'lon')), color = "#d62728")
axs[1].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==2)['ASCAT (Volumetric)'].mean(dim = ('lat', 'lon')), color = "#ff7f0e")
axs[2].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==3)['ASCAT (Volumetric)'].mean(dim = ('lat', 'lon')), color = "#FFE135")
axs[3].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==4)['ASCAT (Volumetric)'].mean(dim = ('lat', 'lon')), color = "#008000")

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_ylim(0, .5)

for ax in axs:
    ax.set_ylabel(r'SM Tavg [$m^3/m^3$]')
    ax.grid(alpha = .3)

for i, ax in enumerate(axs):
    ax.set_title(river_names[i] + ' River')

axs[-1].set_xticklabels(axs[-1].get_xticklabels(), rotation = 45)


fig,axs = plt.subplots(4,1, figsize= (8,8))
axs[0].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==1)['ASCAT (Volumetric)'].std(dim = ('lat', 'lon')), color = "#d62728")
axs[1].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==2)['ASCAT (Volumetric)'].std(dim = ('lat', 'lon')), color = "#ff7f0e")
axs[2].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==3)['ASCAT (Volumetric)'].std(dim = ('lat', 'lon')), color = "#FFE135")
axs[3].plot(da_ASCAT_.time, da_ASCAT_.where(four_river_basin==4)['ASCAT (Volumetric)'].std(dim = ('lat', 'lon')), color = "#008000")

for ax in axs[:-1]:
    ax.set_xticklabels([])
    ax.set_ylim(0, .15)
for i, ax in enumerate(axs):
    ax.set_title(river_names[i] + ' River')

for ax in axs:
    ax.set_ylabel(r'SM Tstd [$m^3/m^3$]')
    ax.grid(alpha = .3)
axs[-1].set_xticklabels(axs[-1].get_xticklabels(), rotation = 45)
