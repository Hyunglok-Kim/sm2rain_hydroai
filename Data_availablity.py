import os
import sys
import platform

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import Point
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

system_name = platform.system()
if system_name == 'Windows': rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': rc('font', family='AppleGothic')
else: rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 2. SETTINGS
# =========================================================
STATION_FP_ASOS = os.path.join(george_FP, 'Precipitation', 'Data', 'Station_ASOS.csv')
STATION_FP_AWS  = os.path.join(george_FP, 'Precipitation', 'Data', 'Station_AWS.csv')
SHP_FILE        = os.path.join(george_FP, 'Precipitation', 'Data', 'Korea.shp')

# Target Analysis Period
TARGET_START = pd.Timestamp('2021-01-01')
TARGET_END   = pd.Timestamp('2025-12-31')
TOTAL_DAYS   = (TARGET_END - TARGET_START).days + 1

print("--- GENERATING AVAILABILITY MAP (ASOS + AWS) ---")

# =========================================================
# 3. LOAD & MERGE STATION DATA
# =========================================================
def load_stations(files):
    dfs = []
    for f in files:
        print(f"Loading {f}...")
        try: df = pd.read_csv(f, encoding='cp949')
        except: df = pd.read_csv(f, encoding='utf-8')
        df.columns = df.columns.str.strip()
        
        # Parse Dates
        df['시작일'] = pd.to_datetime(df['시작일'], errors='coerce')
        df['종료일'] = pd.to_datetime(df['종료일'], errors='coerce')
        
        # Clean ID
        df['지점_clean'] = pd.to_numeric(df['지점'], errors='coerce').fillna(-1).astype(int)
        
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicates (keep last entry if duplicates exist)
    combined = combined.drop_duplicates(subset=['지점_clean'], keep='last')
    return combined

meta_df = load_stations([STATION_FP_ASOS, STATION_FP_AWS])

# =========================================================
# 4. CALCULATE RATIO
# =========================================================
def calculate_ratio(row):
    # 1. Effective Start: Max(Station Start, Target Start)
    s_start = row['시작일'] if pd.notnull(row['시작일']) else pd.Timestamp.min
    eff_start = max(s_start, TARGET_START)
    
    # 2. Effective End: Min(Station End, Target End)
    # If End is NaT (Ongoing), assume it runs until Target End
    s_end = row['종료일'] if pd.notnull(row['종료일']) else TARGET_END
    eff_end = min(s_end, TARGET_END)
    
    # 3. Calculate Overlap
    if eff_end < eff_start:
        return 0.0 # Closed before 2021 or Opened after 2024
    
    overlap_days = (eff_end - eff_start).days + 1
    return overlap_days / TOTAL_DAYS

meta_df['ratio'] = meta_df.apply(calculate_ratio, axis=1)
meta_df['ratio'] = meta_df['ratio'].clip(0, 1)

# Filter out Red/Zero stations (Optional: keep if you want to see them)
active_meta = meta_df[meta_df['ratio'] > 0].copy()

print(f"   - Total Stations: {len(meta_df)}")
print(f"   - Active Stations (Ratio > 0): {len(active_meta)}")

# Create Geometry
points = [Point(xy) for xy in zip(active_meta['경도관서'], active_meta['위도관서'])]
geo_df = gpd.GeoDataFrame(active_meta, geometry=points, crs="EPSG:4326")

# =========================================================
# 5. PLOT MAP
# =========================================================
print("Drawing Map...")

# Load Boundary
boundary_gdf = gpd.read_file(SHP_FILE)
if boundary_gdf.crs is None: boundary_gdf.set_crs("EPSG:5179", inplace=True)
boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(10, 15))

# Draw Korea Background
boundary_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#555555')

# Draw Stations
# vmin=0, vmax=1 ensures Red=0, Green=1
geo_df.plot(
    ax=ax, 
    column='ratio', 
    cmap='RdYlGn', 
    markersize=30,     # Smaller dots for better visibility
    edgecolor='black',
    linewidth=0.3,
    vmin=0, vmax=1,
    legend=False       # Use custom colorbar
)

# [CROP TO MAINLAND]
ax.set_xlim([125.6, 129.6])

# [CUSTOM COLORBAR] - Bottom Horizontal
cax = fig.add_axes([0.15, 0.08, 0.7, 0.02]) 
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
sm._A = []

cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('가용 자료 비율 (2021-2025)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

ax.set_title("기상관측소 가용 자료 비율 (2021-2025)", fontsize=15)
ax.axis('off')

save_name = os.path.join(george_FP, 'Precipitation', 'Figures', "Data_availablity.png")
plt.savefig(save_name, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Done! Saved '{save_name}'")