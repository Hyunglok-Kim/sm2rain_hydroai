from ascat.eumetsat.level2 import AscatL2File
import glob as g

# ASCAT Nat 파일 경로 (.nat)
filename = '/Users/jslee/Desktop/ASCA_SMR_02_M03_20250715032400Z_20250715050558Z_N_O_20250715050635Z/ASCA_SMR_02_M03_20250715032400Z_20250715050558Z_N_O_20250715050635Z.nat'
# filename = '/Users/jslee/Desktop/ASCA_SMR_02_M03_20250715032400Z_20250715050558Z_N_O_20250715050635Z.zip'
# read = ascat.read_native.base.AscatFile(filename)
# read = ascat.read_native.base.AscatFile(filename)
eps_file = AscatL2File(filename)
ds, metadata = eps_file.read(to_xarray=True)
ds

# ds['sm_mean'].plot(x = 'lon', y='lat')

# import matplotlib.pyplot as plt
# plt.scatter(ds['lon'],ds['lat'],c = ds['sm_mean'])