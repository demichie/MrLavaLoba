
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from linecache import getline
from scipy import interpolate
from matplotlib.patches import Ellipse
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
import time
import pickle
import os
import numpy.ma as ma

source = 'BDB_vents_tdm1_1000MIL_020_thickness_masked.asc'

# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(source, i) for i in range(1,7)]
values = [float(h.split(" ")[-1].strip()) \
 for h in hdr]
cols,rows,lx,ly,cell,nd = values
xres = cell
yres = cell * -1

# Load the dem into a numpy array
arr = np.loadtxt(source, skiprows=6)

n, bins, patches = plt.hist(arr, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
plt.draw()


