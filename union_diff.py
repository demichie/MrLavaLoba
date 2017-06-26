import numpy as np
from linecache import getline

source1 = 'HAWAII_020_thickness_masked.asc'
source2 = 'HAWAII_021_thickness_masked.asc'

# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(source1, i) for i in range(1,7)]
values = [float(h.split(" ")[-1].strip()) \
 for h in hdr]
cols,rows,lx,ly,cell,nd = values
xres = cell
yres = cell * -1

# Load the dem into a numpy array
arr = np.loadtxt(source1, skiprows=6)

nx = arr.shape[1]
xs = lx -0.5*cell + np.linspace(0,(nx-1)*cell,nx)
xmin = np.min(xs)
xmax = np.max(xs)

ny = arr.shape[0]
ys = ly+cell*(ny+0.5) - np.linspace(0,(ny-1)*cell,ny)
ymin = np.min(ys)
ymax = np.max(ys)

ys = np.linspace(ymin,ymax,ny)

Zs1 = np.zeros((ny,nx))

Xs,Ys = np.meshgrid(xs,ys)

for i in range(0,ny):

   Zs1[i,0:nx-1] = arr[ny-i-1,0:nx-1]


# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(source2, i) for i in range(1,7)]
values = [float(h.split(" ")[-1].strip()) \
 for h in hdr]
cols,rows,lx,ly,cell,nd = values
xres = cell
yres = cell * -1

# Load the dem into a numpy array
arr = np.loadtxt(source2, skiprows=6)

nx = arr.shape[1]
xs = lx -0.5*cell + np.linspace(0,(nx-1)*cell,nx)
xmin = np.min(xs)
xmax = np.max(xs)

ny = arr.shape[0]
ys = ly+cell*(ny+0.5) - np.linspace(0,(ny-1)*cell,ny)
ymin = np.min(ys)
ymax = np.max(ys)

ys = np.linspace(ymin,ymax,ny)

Zs2 = np.zeros((ny,nx))

Xs,Ys = np.meshgrid(xs,ys)

for i in range(0,ny):

   Zs2[i,0:nx-1] = arr[ny-i-1,0:nx-1]


vol1 = np.sum(Zs1) * cell**2
vol2 = np.sum(Zs2) * cell**2

print('Volume 1',vol1,'Volume 2',vol2)

Zs_union = np.maximum(Zs1,Zs2)

Zs_union = Zs_union / np.maximum(Zs_union,1)
area_union = np.sum(Zs_union) * cell**2

# area_union = np.count_nonzero(Zs_union) * cell**2

Zs_inters = np.minimum(Zs1,Zs2)

Zs_inters = Zs_inters / np.maximum(Zs_inters,1)
area_inters = np.sum(Zs_inters) * cell**2

# area_inters = np.count_nonzero(Zs_inters) * cell**2

fitting_parameter = area_inters/area_union

print('Union area',area_union,'Intersection area',area_inters)
print('Fitting parameter',fitting_parameter)

Zs1_mean = np.mean(Zs1*Zs_inters) * nx*ny / np.count_nonzero(Zs_inters)
Zs2_mean = np.mean(Zs2*Zs_inters) * nx*ny / np.count_nonzero(Zs_inters)

Zs1_vol = Zs1_mean * area_inters
Zs2_vol = Zs2_mean * area_inters

print('Volume 1 in intersection',Zs1_vol,'Volume 2 in intersection',Zs2_vol)

Zs_diff = np.abs(Zs1-Zs2)

Zs_diff = Zs_diff * Zs_inters

avg_thick_diff = np.mean(Zs_diff) * nx*ny / np.count_nonzero(Zs_inters)
std_thick_diff = np.std(Zs_diff) * nx*ny / np.count_nonzero(Zs_inters)
vol_diff = avg_thick_diff * area_inters


rel_err_vol = vol_diff / np.maximum(Zs1_vol,Zs2_vol)

# print('Average thickness difference',avg_thick_diff,'Std thickness difference',std_thick_diff)

print('Thickness relative error',rel_err_vol)











