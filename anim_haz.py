
import matplotlib.pyplot as plt
import numpy as np
from linecache import getline



import time
import os
import numpy.ma as ma
import shutil
import sys


def gen_log_space(limit, n):
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(map(lambda x: round(x)-1, result), dtype=np.uint64)

# Name of the backup file to retrieve the DEM and the outputs
bak_file = 'BDB_vents_tdm1_1000MIL_020_inp.bak'

# This flag select if the coormap used is associated to the values in
# the haz file  or in the masked file
color_with_haz = 0

n_frames = 100

shutil.copy2(bak_file, 'anim_temp.py')

from anim_temp import *

os.remove('anim_temp.py')

run_name = bak_file.replace('_inp.bak','')

haz_file = run_name + '_hazard_full.asc'

masked_file = run_name + '_thickness_full.asc'
dem_file = source

base_name = run_name + '_anim'

print 'DEM file = '+dem_file
print 'haz file = '+haz_file
print 'masked file = '+masked_file
print 'base name = '+base_name

# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(dem_file, i) for i in range(1,7)]
values = [float(h.split(" ")[-1].strip()) \
 for h in hdr]
cols,rows,lx,ly,cell,nd = values
xres = cell
yres = cell * -1

# Load the dem into a numpy array
arr = np.loadtxt(dem_file, skiprows=6)

# Load the haz file into a numpy array
arr2 = np.loadtxt(haz_file, skiprows=6)

if ( color_with_haz == 0 ):
   # Load the output into a numpy array
   masked_output = np.loadtxt(masked_file, skiprows=6)

else:

   masked_output = arr2


nx = arr.shape[1]
xs = lx -0.5*cell + np.linspace(0,(nx-1)*cell,nx)
xmin = np.min(xs)
xmax = np.max(xs)

ny = arr.shape[0]
ys = ly+cell*(ny+0.5) - np.linspace(0,(ny-1)*cell,ny)
ymin = np.min(ys)
ymax = np.max(ys)

ys = np.linspace(ymin,ymax,ny)

topo = np.zeros((ny,nx))
haz = np.zeros((ny,nx))
masked_data = np.zeros((ny,nx))

Xs,Ys = np.meshgrid(xs,ys)

for i in range(0,ny):

   topo[i,0:nx-1] = arr[ny-i-1,0:nx-1]
   masked_data[i,0:nx-1] = masked_output[ny-i-1,0:nx-1]
   haz[i,0:nx-1] = arr2[ny-i-1,0:nx-1]



haz_min = 1
haz_max = np.int(np.max(haz))
data_max = np.int(np.max(masked_data))

print 'Number of frames = ',n_frames

masked_zeros = ma.masked_where(haz < 1, masked_data)

image_data = ma.masked_where( (haz > 1) , masked_zeros)


fig = plt.figure()
plt.contour(Xs, Ys, topo ,150,zorder=1,cmap=plt.get_cmap('summer'))
plt.axis('equal')
plt.ylim([ymin,ymax])
plt.xlim([xmin,xmax])
plt.ion()
plt.show()
plt.draw()


image = plt.imshow(image_data,
    cmap=plt.get_cmap('hot'),
    vmin=1, vmax=data_max,
    origin='lower', extent=[xmin,xmax,ymin,ymax],
    zorder=10)

frame_name = base_name + '_{0:03}'.format(1) + '.png'
plt.savefig(frame_name,dpi=200)


# for i in range(2,dist_max):

frame = 0

x = gen_log_space(haz_max, n_frames)

for j in reversed(x):

   i = np.rint(j)

   last_percentage = np.rint(i*20.0/(haz_max-1))
   integer_value =  np.rint(i*100.0/(haz_max-1))
   sys.stdout.write('\r')
   sys.stdout.write("[%-20s] %d%%" % ('='*(last_percentage), integer_value))
   sys.stdout.flush()

   image_data = ma.masked_where( (haz < i) , masked_zeros)

   image.set_data(image_data)

   frame_name = base_name + '_{0:03}'.format(frame) + '.png'
   plt.savefig(frame_name,dpi=200)
   plt.draw()
   frame = frame + 1


plt.ioff()
plt.show()

