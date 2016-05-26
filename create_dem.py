# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from linecache import getline
from scipy import interpolate
from scipy.stats import beta
from matplotlib.patches import Ellipse
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import time
import os
import numpy.ma as ma
import sys
import shutil
import datetime
from mpl_toolkits.mplot3d import Axes3D


x_domain = 5000 # x-extension of the domain (in meters)
y_domain = 5000 # y-extension of the domain (in meters)

DEM_slope = 10.0 # slope of the topography

# Parameters of the channels: the total widht is given by width+2*x_flank

width = 10    # width of the base of the channel
x_flank = 100.0  # width of the flanks of the channel
depth = 20.0  # depth of the channel

# cell size
cell = 20 



# convert to radiants
DEM_slope_rad = DEM_slope/180*np.pi

# build the X,Y grid
xll = -0.5 * x_domain
yll = -0.5 * y_domain

xtr = 0.5 * x_domain
ytr = 0.5 * y_domain

x_grid = np.arange(xll,xtr,cell)
y_grid = np.arange(yll,ytr,cell)

nx = len(x_grid)
ny = len(y_grid)

X = np.zeros((nx,ny))
Y = np.zeros((nx,ny))
Z = np.zeros((nx,ny))

elev = np.zeros((nx,ny))
deltaZ = np.zeros((nx,ny))


for i in range(0,nx):

    elev[i,0:ny] = x_grid[i] * np.tan(DEM_slope_rad)

    X[i,0:ny] = x_grid[i]

for j in range(0,ny):

    Y[0:nx,j] = y_grid[j]

    if ( np.abs(y_grid[j]) < width ):

        deltaZ[0:nx,j] = - depth

    elif (np.abs(y_grid[j]) < ( width + x_flank ) ):

        arg_cos = np.pi * ( width + x_flank - np.abs(y_grid[j]) ) / x_flank 

        deltaZ[0:nx,j] = 0.5 * ( np.cos(arg_cos) - 1.0 ) * depth

    else:

        deltaZ[0:nx,j] = 0

Z = elev + deltaZ

header = "ncols     %s\n" % nx
header += "nrows    %s\n" % ny
header += "xllcorner " + str(xll) +"\n"
header += "yllcorner " + str(yll) +"\n"
header += "cellsize " + str(cell) +"\n"
header += "NODATA_value -9999\n"

dem_file = 'synth_DEM.asc'
        
np.savetxt(dem_file, np.transpose(Z), header=header, fmt='%1.5f',comments='')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.auto_scale_xyz
ax.plot_surface(X, Y, Z,  rstride=4, cstride=4, color='b')

plt.show()

