import os
import sys
import platform

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.prepared import prep
import platform
from matplotlib import font_manager, rc

if platform.system() == 'Darwin': # Mac Mini
    base_FP = '/Users/geobambus'
    cpuserver_data_FP = os.path.join(base_FP, 'cpuserver_data')
    george_FP = os.path.join(cpuserver_data_FP, 'python_modules', 'kunhee')
elif platform.system() == 'Linux': # Workstation
    base_FP = '/home/kunhee'
    if os.path.exists(base_FP):
        cpuserver_data_FP = os.path.join(base_FP, 'cpuserver_data')
        nas_FP = os.path.join(base_FP, 'NAS')
        das_FP = os.path.join(base_FP, 'DAS')
        george_FP = os.path.join(cpuserver_data_FP, 'python_modules', 'kunhee')
    else: # CPU Server
        base_FP = '/data'
        nas_FP = '/data'
        das_FP = '/data'
        cpuserver_data_FP = '/data'

# 1. FONT SETTINGS
system_name = platform.system()
if system_name == 'Windows': rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': rc('font', family='AppleGothic')
else: rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. SETTINGS
TARGET_DATE = '2023-07-14'
SHP_FILE    = os.path.join(george_FP, 'Precipitation', 'Data', 'Korea.shp')
RESOLUTION  = 0.05 
STEP        = 100 

# File Paths
STATION_FP_ASOS = os.path.join(george_FP, 'Precipitation', 'Data', 'Station_ASOS.csv')
STATION_FP_AWS  = os.path.join(george_FP, 'Precipitation', 'Data', 'Station_AWS.csv')
DATA_FP_ASOS    = os.path.join(george_FP, 'Precipitation', 'Data', 'Data_ASOS.csv')
DATA_FP_AWS     = os.path.join(george_FP, 'Precipitation', 'Data', 'Data_AWS.csv')

print(f"--- IDW MAPPING (ASOS + AWS) FOR {TARGET_DATE} ---")

# 3. LOAD & MERGE DATA
def load_stations(files):
    dfs = []
    for f in files:
        try: df = pd.read_csv(f, encoding='cp949')
        except: df = pd.read_csv(f, encoding='utf-8')
        df.columns = df.columns.str.strip()
        df['지점_clean'] = pd.to_numeric(df['지점'], errors='coerce').fillna(-1).astype(int)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['지점_clean'], keep='last')
    return combined

def load_rain(files, target_date):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df['일시'] = pd.to_datetime(df['일시'])
        df['지점_clean'] = pd.to_numeric(df['지점'], errors='coerce').fillna(-1).astype(int)
        
        # Filter for Target Date
        day_df = df[df['일시'].dt.date == pd.to_datetime(target_date).date()]
        dfs.append(day_df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # [LOGIC] This works for BOTH Hourly and Daily data
    # Hourly -> Sums 24 rows
    # Daily  -> Sums 1 row (No change)
    daily_sum = combined.groupby('지점_clean')['강수량'].sum(min_count=0).reset_index()
    daily_sum['강수량'] = daily_sum['강수량'].fillna(0)
    return daily_sum

print("1. Loading and Merging Data...")
meta_df = load_stations([STATION_FP_ASOS, STATION_FP_AWS])
rain_df = load_rain([DATA_FP_ASOS, DATA_FP_AWS], TARGET_DATE)

merged_df = pd.merge(meta_df, rain_df, on='지점_clean', how='inner')
print(f"   - Total Matched Stations: {len(merged_df)}")

x_obs = merged_df['경도관서'].values
y_obs = merged_df['위도관서'].values
z_obs = merged_df['강수량'].values

# 4. IDW INTERPOLATION
boundary_gdf = gpd.read_file(SHP_FILE)
if boundary_gdf.crs is None: boundary_gdf.set_crs("EPSG:5179", inplace=True)
boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

minx, miny, maxx, maxy = boundary_gdf.total_bounds
grid_x = np.arange(minx, maxx, RESOLUTION)
grid_y = np.arange(miny, maxy, RESOLUTION)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

print("2. Interpolating...")
grid_z = griddata((x_obs, y_obs), z_obs, (grid_x, grid_y), method='cubic', fill_value=np.nan)
grid_z_nearest = griddata((x_obs, y_obs), z_obs, (grid_x, grid_y), method='nearest')
mask = np.isnan(grid_z)
grid_z[mask] = grid_z_nearest[mask]

# Clip
try: boundary_poly = boundary_gdf.union_all()
except: boundary_poly = boundary_gdf.unary_union
prepared_poly = prep(boundary_poly)
rows, cols = grid_z.shape
for r in range(rows):
    for c in range(cols):
        if not prepared_poly.contains(Point(grid_x[r, c], grid_y[r, c])):
            grid_z[r, c] = np.nan

# 5. PLOT
grid_z[grid_z < 0] = 0
max_val = np.nanmax(grid_z)
if max_val == 0: max_val = 1
nice_max = math.ceil(max_val / STEP) * STEP
levels = np.linspace(0, nice_max, 51) 

fig, ax = plt.subplots(figsize=(10, 15))
boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

contour = ax.contourf(
    grid_x, grid_y, grid_z, 
    levels=levels, 
    cmap='Blues', 
    alpha=0.8,
    # extend='max'  <--- REMOVED (Square Ends)
)
ax.scatter(x_obs, y_obs, c='red', s=10, edgecolors='none', alpha=0.5, label='Stations')

# Crop to Mainland
ax.set_xlim([125.6, 129.6])

# [CUSTOM COLORBAR] - Bottom Horizontal & Square Ends
cax = fig.add_axes([0.15, 0.08, 0.7, 0.02]) 
cbar = plt.colorbar(contour, cax=cax, orientation='horizontal', ticks=range(0, nice_max+1, STEP))
cbar.set_label('일강수량 (mm)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

ax.set_title(f"역거리 가중법 기반 일강수량 지도 ({TARGET_DATE})", fontsize=15)
ax.axis('off')

save_name = os.path.join(george_FP, 'Precipitation', 'Figures', f"IDW_Map_{TARGET_DATE}.png")
plt.savefig(save_name, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Done! Saved {save_name}")