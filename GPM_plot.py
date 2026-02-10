import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import h5py 
import platform
import pandas as pd
from matplotlib import font_manager, rc
import math

# =========================================================
# 1. SETTINGS
# =========================================================
system_name = platform.system()
if system_name == 'Windows': rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': rc('font', family='AppleGothic')
else: rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

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

SHP_FILE = os.path.join(george_FP, 'Precipitation', 'Data', 'Korea.shp')

# Target: 2023-07-14 KST
TARGET_DATE_KST = '2023-07-14'
BBOX = [33.1, 38.7, 125.6, 129.7] 
STEP = 100 

print(f"--- GPM MAPPING FOR {TARGET_DATE_KST} (KST Adjusted) ---")

# =========================================================
# 2. TIME ZONE CALCULATION (KST -> UTC)
# =========================================================
# KST Day: 00:00:00 to 23:59:59
start_kst = pd.Timestamp(f"{TARGET_DATE_KST} 00:00:00")
end_kst   = pd.Timestamp(f"{TARGET_DATE_KST} 23:59:59")

# Convert to UTC (KST - 9 hours)
start_utc = start_kst - pd.Timedelta(hours=9)
end_utc   = end_kst   - pd.Timedelta(hours=9)

print(f"1. Time Range:")
print(f"   KST: {start_kst} ~ {end_kst}")
print(f"   UTC: {start_utc} ~ {end_utc} (Files needed from these times)")

# We need files from TWO folders:
# 1. The day of 'start_utc' (e.g., July 13)
# 2. The day of 'end_utc'   (e.g., July 14)
days_to_check = pd.date_range(start_utc.date(), end_utc.date(), freq='D')

# =========================================================
# 3. GATHER FILES
# =========================================================
valid_files = []

for d in days_to_check:
    y_str = d.strftime('%Y')
    m_str = d.strftime('%m')
    d_str = d.strftime('%d')
    
    folder_path = os.path.join(nas_FP, 'GPM', y_str, m_str, d_str)
    
    # Grab all HDF5
    raw_files = sorted(glob.glob(os.path.join(folder_path, '*.HDF5')))
    if not raw_files:
        raw_files = sorted(glob.glob(os.path.join(folder_path, '*.h5')))
        
    # Check Timestamp in filename
    # Filename format example: 3B-HHR...20230713-S150000...
    for fpath in raw_files:
        fname = os.path.basename(fpath)
        try:
            # Extract Date/Time part (Assuming standard IMERG name)
            # Find part starting with '20' (Year)
            # 3B-HHR.MS.MRG.3IMERG.20230713-S150000-E152959.0900.V07B.HDF5
            parts = fname.split('.')
            date_part = [p for p in parts if p.startswith('20') and '-S' in p][0]
            
            # Parse '20230713-S150000'
            date_str = date_part.split('-S')[0] # 20230713
            time_str = date_part.split('-S')[1].split('-E')[0] # 150000
            
            file_dt = pd.to_datetime(f"{date_str} {time_str}", format='%Y%m%d %H%M%S')
            
            # Filter: Is this file inside our UTC target window?
            if start_utc <= file_dt <= end_utc:
                valid_files.append(fpath)
                
        except:
            # If filename format is weird, just skip or print warning
            pass

print(f"   - Found {len(valid_files)} valid files matching the KST window.")
if len(valid_files) < 40:
    print("   [WARNING] File count is low (Expected ~48). Totals might be low.")

# =========================================================
# 4. READ & ACCUMULATE
# =========================================================
daily_rain = None
lats = None
lons = None

print("2. Accumulating Rainfall...")

for fpath in valid_files:
    try:
        with h5py.File(fpath, 'r') as f:
            # 1. Get Coordinates (Once)
            if lats is None:
                # Try standard paths
                grp = f['Grid'] if 'Grid' in f else (f['S1'] if 'S1' in f else f)
                raw_lats = grp['lat'][:]
                raw_lons = grp['lon'][:]
                
                # BBox Optimization
                lat_idx = np.where((raw_lats >= BBOX[0]) & (raw_lats <= BBOX[1]))[0]
                lon_idx = np.where((raw_lons >= BBOX[2]) & (raw_lons <= BBOX[3]))[0]
                
                lats = raw_lats[lat_idx]
                lons = raw_lons[lon_idx]
                daily_rain = np.zeros((len(lon_idx), len(lat_idx)))

            # 2. Get Precip
            if 'Grid' in f and 'precipitationCal' in f['Grid']:
                precip_raw = f['Grid']['precipitationCal']
            elif 'Grid' in f and 'precipitation' in f['Grid']:
                precip_raw = f['Grid']['precipitation']
            else:
                continue

            # Subset
            if len(precip_raw.shape) == 2:
                precip_subset = precip_raw[lon_idx.min():lon_idx.max()+1, lat_idx.min():lat_idx.max()+1]
            elif len(precip_raw.shape) == 3:
                precip_subset = precip_raw[0, lon_idx.min():lon_idx.max()+1, lat_idx.min():lat_idx.max()+1]
            
            precip_subset[precip_subset < 0] = 0
            
            # Accumulate (mm/hr * 0.5h)
            daily_rain += (precip_subset * 0.5)

    except Exception as e:
        print(f"   Error reading {os.path.basename(fpath)}: {e}")

# Transpose for Plotting (GPM is Lon x Lat -> Need Lat x Lon for some plots, or align with meshgrid)
# Meshgrid (lons, lats) creates X(Lat, Lon) Y(Lat, Lon)? No.
# meshgrid(x, y) -> X has shape (len(y), len(x)). 
# daily_rain is (len(lon), len(lat)).
# So daily_rain.T is (len(lat), len(lon)), which matches meshgrid shape.
grid_lon, grid_lat = np.meshgrid(lons, lats)
final_rain = daily_rain.T 

# =========================================================
# 5. PLOT
# =========================================================
print("3. Plotting...")
boundary_gdf = gpd.read_file(SHP_FILE)
if boundary_gdf.crs is None: boundary_gdf.set_crs("EPSG:5179", inplace=True)
boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

# Determine Max for Scale (Using same STEP logic)
max_val = np.nanmax(final_rain)
if max_val == 0: max_val = 1
nice_max = math.ceil(max_val / STEP) * STEP
levels = np.linspace(0, nice_max, 51) 

fig, ax = plt.subplots(figsize=(10, 15))

# Plot GPM
contour = ax.contourf(
    grid_lon, grid_lat, final_rain,
    levels=levels,
    cmap='Blues',
    alpha=0.9
)

boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

# Crop
ax.set_xlim([BBOX[2], BBOX[3]])
ax.set_ylim([BBOX[0], BBOX[1]])

# Bottom Colorbar
cax = fig.add_axes([0.15, 0.08, 0.7, 0.02]) 
cbar = plt.colorbar(contour, cax=cax, orientation='horizontal', ticks=range(0, nice_max+1, STEP))
cbar.set_label('일강수량 (mm)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

ax.set_title(f"GPM 위성 지도 ({TARGET_DATE_KST})", fontsize=15)
ax.axis('off')

save_name = os.path.join(george_FP, 'Precipitation', 'Figures', f"GPM_Map_{TARGET_DATE_KST}.png")
plt.savefig(save_name, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Done! Saved {save_name}")
print(f"Max GPM Value Found: {max_val:.2f} mm")