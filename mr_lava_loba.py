from input_data_advanced import n_check_loop
import numpy as np
from linecache import getline
from scipy.stats import beta
import time
import os
import numpy.ma as ma
import sys
import shutil
import datetime
import rtnorm
from random import randrange
from os.path import exists
import gc
import pandas as pd

from input_data import run_name
from input_data import source
from input_data import x_vent
from input_data import y_vent
from input_data import hazard_flag
from input_data import masking_threshold
from input_data import n_flows
from input_data import min_n_lobes
from input_data import max_n_lobes
from input_data import thickening_parameter
from input_data import lobe_area
from input_data import inertial_exponent
from input_data import lobe_exponent
from input_data import max_slope_prob
from input_data import thickness_ratio
from input_data import fixed_dimension_flag
from input_data import vent_flag
from input_data_advanced import npoints
from input_data_advanced import n_init
from input_data_advanced import dist_fact
from input_data_advanced import flag_threshold
from input_data_advanced import a_beta
from input_data_advanced import b_beta
from input_data_advanced import max_aspect_ratio
from input_data_advanced import saveraster_flag
from input_data_advanced import aspect_ratio_coeff
from input_data_advanced import start_from_dist_flag

from input_data_advanced import force_max_length
if force_max_length:

    from input_data_advanced import max_length


def interp2Dgrids(xin, yin, Zin, Xout, Yout):
    """
    Interpolation from a regular grid to a second regular grid

    @params:
        xin      - Required : original grid X values (1D Dble)
        yin      - Required : original grid Y values (1D Dble)
        Zin      - Required : original grid Z values (2D Dble)
        xout     - Required : new grid X values (2D Dble)
        yout     - Required : new grid Y values (2D Dble)
    """
    xinMin = np.min(xin)

    yinMin = np.min(yin)

    cellin = xin[1] - xin[0]

    if (Xout.ndim == 2):

        xout = Xout[0, :]

    else:

        xout = Xout

    if (Yout.ndim == 2):

        yout = Yout[:, 0]

    else:

        yout = Yout

    # Search for the cell containing the center of the parent lobe
    xi = (xout - xinMin) / cellin
    yi = (yout - yinMin) / cellin

    # Indexes of the lower-left corner of the cell containing the center of
    # the parent lobe
    ix = np.maximum(0, np.minimum(xin.shape[0] - 2, np.floor(xi).astype(int)))
    iy = np.maximum(0, np.minimum(yin.shape[0] - 2, np.floor(yi).astype(int)))

    # Indexes of the top-right corner of the cell containing the center of
    # the parent lobe
    ix1 = ix + 1
    iy1 = iy + 1

    # Relative coordinates of the center of the parent lobe in the cell

    xi_fract = np.maximum(0.0,
                          np.minimum(1.0, (xi - ix).reshape(1, Xout.shape[1])))

    yi_fract = np.maximum(0.0,
                          np.minimum(1.0, (yi - iy).reshape(Yout.shape[0], 1)))

    xi_out_yi = np.outer(yi_fract, xi_fract)

    Zout = xi_out_yi * Zin[np.ix_(iy1, ix1)] + \
        (xi_fract - xi_out_yi) * Zin[np.ix_(iy, ix1)] + \
        (yi_fract - xi_out_yi) * Zin[np.ix_(iy1, ix)] + \
        (1.0 - xi_fract - yi_fract + xi_out_yi) * Zin[np.ix_(iy, ix)]

    return Zout


def ellipse(xc, yc, ax1, ax2, angle, X_circle, Y_circle):

    cos_angle = np.cos(angle * np.pi / 180)
    sin_angle = np.sin(angle * np.pi / 180)

    # x1 = xc + ax1 * cos_angle
    # y1 = yc + ax1 * sin_angle

    # x2 = xc - ax2 * sin_angle
    # y2 = yc + ax2 * cos_angle

    X = ax1 * X_circle
    Y = ax2 * Y_circle

    xe = xc + X * cos_angle - Y * sin_angle
    ye = yc + X * sin_angle + Y * cos_angle

    return (xe, ye)


def local_intersection(Xc_local, Yc_local, xc_e, yc_e, ax1, ax2, angle, xv, yv,
                       nv2):

    # the accuracy of this procedure depends on the resolution of xv and yv
    # representing a grid of points [-0.5*cell;0.5*cell] X [-0.5*cell;0.5*cell]
    # built around the centers

    nx_cell = Xc_local.shape[0]
    ny_cell = Xc_local.shape[1]

    c = np.cos(angle * np.pi / 180)
    s = np.sin(angle * np.pi / 180)

    c1 = c / ax1
    s1 = s / ax1

    c2 = c / ax2
    s2 = s / ax2

    xv = xv - xc_e
    yv = yv - yc_e

    Xc_local_1d = Xc_local.ravel()
    Yc_local_1d = Yc_local.ravel()

    c1xv_p_s1yv = c1 * xv + s1 * yv
    c2yv_m_s2yv = c2 * yv - s2 * xv

    term1 = (c1**2 + s2**2) * Xc_local_1d**2
    term2 = (2 * c1 * s1 - 2 * c2 * s2) * Xc_local_1d * Yc_local_1d
    term3 = np.tensordot(Xc_local_1d,
                         2 * c1 * c1xv_p_s1yv - 2 * s2 * c2yv_m_s2yv, 0)
    term4 = (c2**2 + s1**2) * Yc_local_1d**2
    term5 = np.tensordot(Yc_local_1d,
                         2 * c2 * c2yv_m_s2yv + 2 * s1 * c1xv_p_s1yv, 0)
    term6 = c1xv_p_s1yv**2 + c2yv_m_s2yv**2

    term124 = term1 + term2 + term4
    term356 = term3 + term5 + term6

    term_tot = term124 + term356.transpose()

    inside = (term_tot <= 1)

    area_fract_1d = np.sum(inside.astype(float), axis=0)

    area_fract_1d /= nv2

    area_fract = area_fract_1d.reshape(nx_cell, ny_cell)

    return (area_fract)


# Main start here

print("")
print("Mr Lava Loba by M.de' Michieli Vitturi and S.Tarquini")
print("")

# read the run parameters form the file inpot_data.py

n_vents = len(x_vent)

try:

    from input_data import x_vent_end

except ImportError:

    print('x_vent_end not used')

try:

    from input_data import y_vent_end

except ImportError:

    print('y_vent_end not used')

if (('x_vent_end' in globals()) and (len(x_vent_end) > 0) and (vent_flag > 3)):

    first_j = 0
    cum_fiss_length = np.zeros(n_vents + 1)

else:

    first_j = 1
    cum_fiss_length = np.zeros(n_vents)

for j in range(first_j, n_vents):

    if (('x_vent_end' in globals()) and (len(x_vent_end) > 0)
            and (vent_flag > 3)):

        delta_xvent = x_vent_end[j] - x_vent[j]
        delta_yvent = y_vent_end[j] - y_vent[j]

        cum_fiss_length[j+1] = cum_fiss_length[j] + \
            np.sqrt(delta_xvent**2 + delta_yvent**2)

    else:

        delta_xvent = x_vent[j] - x_vent[j - 1]
        delta_yvent = y_vent[j] - y_vent[j - 1]

        cum_fiss_length[j] = cum_fiss_length[j-1] + \
            np.sqrt(delta_xvent**2 + delta_yvent**2)

try:

    from input_data import fissure_probabilities

except ImportError:

    print('fissure_probabilities not used')

if ('fissure_probabilities' in globals()):

    if (vent_flag == 8):

        cum_fiss_length = np.cumsum(fissure_probabilities)

    elif (vent_flag > 5):

        cum_fiss_length[1:] = np.cumsum(fissure_probabilities)

if (n_vents > 1):
    cum_fiss_length = cum_fiss_length.astype(float) / cum_fiss_length[-1]

# print(cum_fiss_length)

# search if another run with the same base name already exists
i = 0

condition = True

base_name = run_name

while condition:

    run_name = base_name + '_{0:03}'.format(i)

    backup_advanced_file = run_name + '_advanced_inp.bak'
    backup_file = run_name + '_inp.bak'

    condition = os.path.isfile(backup_file)

    i = i + 1

# create a backup file of the input parameters
shutil.copy2('input_data_advanced.py', backup_advanced_file)
shutil.copy2('input_data.py', backup_file)

print('Run name', run_name)
print('')

if (a_beta == 0) and (b_beta == 0):

    alloc_n_lobes = int(max_n_lobes)

else:

    x_beta = np.rint(range(0, n_flows)) / (n_flows - 1)

    beta_pdf = beta.pdf(x_beta, a_beta, b_beta)

    alloc_n_lobes = int(
        np.rint(min_n_lobes + 0.5 *
                (max_n_lobes - min_n_lobes) * np.max(beta_pdf)))

    print('Flow with the maximum number of lobes', np.argmax(beta_pdf))

print('Maximum number of lobes', alloc_n_lobes)

# initialize the arrays for the lobes variables
angle = np.zeros(alloc_n_lobes)
x = np.zeros(alloc_n_lobes)
y = np.zeros(alloc_n_lobes)
x1 = np.zeros(alloc_n_lobes)
x2 = np.zeros(alloc_n_lobes)
h = np.zeros(alloc_n_lobes)

dist_int = np.zeros(alloc_n_lobes, dtype=int) - 1
descendents = np.zeros(alloc_n_lobes, dtype=int)
parent = np.zeros(alloc_n_lobes, dtype=int)
alfa_inertial = np.zeros(alloc_n_lobes)

try:

    from input_data import volume_flag

except ImportError:

    print('volume_flag non specified in input')
    sys.exit()

if (volume_flag == 1):

    try:

        from input_data import total_volume

    except ImportError:

        print('total_volume needed')
        sys.exit()

    if (fixed_dimension_flag == 1):

        avg_lobe_thickness = total_volume / \
            (n_flows * lobe_area * 0.5 * (min_n_lobes + max_n_lobes))
        sys.stdout.write("Average Lobe thickness = %f m\n\n" %
                         (avg_lobe_thickness))

    elif (fixed_dimension_flag == 2):

        lobe_area = total_volume / \
            (n_flows * avg_lobe_thickness * 0.5 * (min_n_lobes + max_n_lobes))
        sys.stdout.write("Lobe area = %f m2\n\n" % (lobe_area))

# Needed for numpy conversions
deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

# Define variables needed to build the ellipses
t = np.linspace(0.0, 2.0 * np.pi, npoints)
X_circle = np.cos(t)
Y_circle = np.sin(t)

# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(source, i) for i in range(1, 7)]

values = [float(h.split(" ")[-1].strip()) for h in hdr]
del hdr

cols, rows, lx, ly, cell, nd = values
# del values

cols = int(cols)
rows = int(rows)

try:

    from input_data import west_to_vent

except ImportError:

    print("west_to_vent not defined in input file")

try:

    from input_data import east_to_vent

except ImportError:

    print("east_to_vent not defined in input file")

try:

    from input_data import south_to_vent

except ImportError:

    print("south_to_vent not defined in input file")

try:

    from input_data import north_to_vent

except ImportError:

    print("north_to_vent not defined in input file")

crop_flag = ("west_to_vent" in locals()) and ("east_to_vent" in locals()) \
    and ("south_to_vent" in locals()) and ("north_to_vent" in locals())

print('west_to_vent', west_to_vent)
print('x_vent', x_vent)
print('Crop flag = ', crop_flag)

if sys.version_info >= (3, 0):
    start = time.process_time()
else:
    start = time.clock()

source_npy = source.replace(".asc", ".npy")

if os.path.isfile(source_npy):
    print(source_npy, " exists")
else:
    print(source_npy, " does not exist")
    data = np.loadtxt(source, skiprows=6)
    np.save(source_npy, data)
    del data

if crop_flag:

    # Load the dem into a numpy array
    arr_temp = np.flipud(np.load(source_npy))

    # the values are associated to the center of the pixels
    xc_temp = lx + cell * (0.5 + np.arange(0, arr_temp.shape[1]))
    yc_temp = ly + cell * (0.5 + np.arange(0, arr_temp.shape[0]))

    xW = np.min(x_vent) - west_to_vent
    xE = np.max(x_vent) + east_to_vent
    yS = np.min(y_vent) - south_to_vent
    yN = np.max(y_vent) + north_to_vent

    # crop the DEM to the desired domain
    iW = np.maximum(0, (np.floor((xW - lx) / cell)).astype(int))
    iE = np.minimum(cols, (np.ceil((xE - lx) / cell)).astype(int))
    jS = np.maximum(0, (np.floor((yS - ly) / cell)).astype(int))
    jN = np.minimum(rows, (np.ceil((yN - ly) / cell)).astype(int))

    print('Cropping of original DEM')
    print('xW,xE,yS,yN', xW, xE, yS, yN)
    print('iW,iE,jS,jN', iW, iE, jS, jN)
    print('')

    arr = arr_temp[jS:jN, iW:iE]
    xc = xc_temp[iW:iE]
    yc = yc_temp[jS:jN]

    lx = xc[0] - 0.5 * cell
    ly = yc[0] - 0.5 * cell

    nx = arr.shape[1]
    ny = arr.shape[0]

    header = "ncols     %s\n" % arr.shape[1]
    header += "nrows    %s\n" % arr.shape[0]
    header += "xllcorner " + str(lx) + "\n"
    header += "yllcorner " + str(ly) + "\n"
    header += "cellsize " + str(cell) + "\n"
    header += "NODATA_value " + str(nd) + "\n"

    output_DEM = run_name + '_DEM.asc'

    np.savetxt(output_DEM,
               np.flipud(arr),
               header=header,
               fmt='%1.5f',
               comments='')

    del arr_temp
    del xc_temp
    del yc_temp
    gc.collect()

else:

    # Load the dem into a numpy array
    arr = np.flipud(np.load(source_npy))

    nx = arr.shape[1]
    ny = arr.shape[0]

    # the values are associated to the center of the pixels
    xc = lx + cell * (0.5 + np.arange(0, nx))
    yc = ly + cell * (0.5 + np.arange(0, ny))

gc.collect()

if sys.version_info >= (3, 0):
    elapsed = (time.process_time() - start)
else:
    elapsed = (time.clock() - start)

print('Time to read DEM ' + str(elapsed) + 's')

xcmin = np.min(xc)
xcmax = np.max(xc)

ycmin = np.min(yc)
ycmax = np.max(yc)

Xc, Yc = np.meshgrid(xc, yc)

Zc = np.zeros((ny, nx))
np.copyto(Zc, arr)

filling_parameter = (1.0 - thickening_parameter) * np.ones_like(Zc)

try:

    from input_data import channel_file
    from input_data import alfa_channel
    from input_data import d1
    from input_data import d2
    from input_data import eps

    check_channel_file = exists(channel_file)

except ImportError:

    print('Channel parameters not defined:')
    print('- channel_file')
    print('- d1')
    print('- d2')
    print('- eps')
    print('- alfa_chaneel')

    check_channel_file = False
    alfa_channel = 0.0

if check_channel_file:

    import shapefile
    from shapely.geometry import Point, LineString, MultiPoint
    from shapely.ops import nearest_points

    # arrays of components of the direction vectors computer from channel
    vx = np.zeros_like(Xc)
    vy = np.zeros_like(Yc)

    # arrays of distances from channel
    distxy = np.zeros_like(Yc)

    print('')

    print('Reading shapefile ' + channel_file)

    sf = shapefile.Reader(channel_file)

    shapes = sf.shapes()
    shapeRecs = sf.shapeRecords()
    points = shapeRecs[0].shape.points[0:]
    ln = LineString(points)

    pnew_x = points[-1][0] + 200.0 * (points[-1][0] - points[-2][0])
    pnew_y = points[-1][1] + 200.0 * (points[-1][1] - points[-2][1])

    points.append([pnew_x, pnew_y])

    nlx = []
    nly = []
    for i in range(len(points) - 1):
        nnx = points[i + 1][0] - points[i][0]
        nny = points[i + 1][1] - points[i][1]
        nn = np.sqrt(nnx**2 + nny**2)
        nlx.append(nnx / nn)
        nly.append(nny / nn)
        # print(nlx[i],nly[i])

    minx, miny, maxx, maxy = ln.bounds

    print('Channel Bounding Box', minx, miny, maxx, maxy)

    dx = 3.0 * d2

    minx = minx - dx
    maxx = maxx + dx
    miny = miny - dx
    maxy = maxy + dx

    min_xe = minx
    max_xe = maxx

    min_ye = miny
    max_ye = maxy

    xi = (min_xe - xcmin) / cell
    ix = np.floor(xi)
    i_left = ix.astype(int)

    xi = (max_xe - xcmin) / cell
    ix = np.floor(xi)
    i_right = ix.astype(int) + 2

    yj = (min_ye - ycmin) / cell
    jy = np.floor(yj)
    j_bottom = jy.astype(int)

    yj = (max_ye - ycmin) / cell
    jy = np.floor(yj)
    j_top = jy.astype(int) + 2

    print('i_left,i_right', i_left, i_right)
    print('j_bottom,j_top', j_bottom, j_top)

    # define the subgrid of pixels to check for coverage
    Xgrid = Xc[j_bottom:j_top, i_left:i_right]
    Ygrid = Yc[j_bottom:j_top, i_left:i_right]

    xgrid = Xgrid[0, :]
    ygrid = Ygrid[:, 0]

    coords = np.vstack((Xgrid.ravel(), Ygrid.ravel())).T
    pts = MultiPoint(coords)
    """
    dist_pl = np.zeros_like(arr)

    for idx, valx in enumerate(xgrid):
        for idy, valy in enumerate(ygrid):

            pt = Point(valx,valy)
            dist = np.exp(-pt.distance(ln)**2 / ( 2.0*d1**2))
            dist_pl[j_bottom+idy,i_left+idx] = dist
            filling_parameter[j_bottom+idy,i_left+idx] *= ( 1.0 - dist)


    header = "ncols     %s\n" % arr.shape[1]
    header += "nrows    %s\n" % arr.shape[0]
    header += "xllcorner " + str(lx) +"\n"
    header += "yllcorner " + str(ly) +"\n"
    header += "cellsize " + str(cell) +"\n"
    header += "NODATA_value 0\n"

    print('dist_pl',np.shape(dist_pl))

    output_full = run_name + '_channel_distance.asc'
    Zc -= 10*dist_pl


    print('Zc',np.shape(Zc))
    alfa_channel = 0.0

    np.savetxt(output_full, np.flipud(dist_pl), header=header,
               fmt='%1.5f',comments='')
    """
    # print(ciao)

    for idx, valx in enumerate(xgrid):

        for idy, valy in enumerate(ygrid):

            pt = Point(valx, valy)

            p1, p2 = nearest_points(ln, pt)
            xx, yy = p1.coords.xy
            vx1 = xx - valx
            vy1 = yy - valy
            v1mod = np.sqrt(vx1**2 + vy1**2)
            vx1 = vx1 / v1mod
            vy1 = vy1 / v1mod

            dist = []
            for i in range(len(points) - 1):

                dist.append(
                    np.maximum(eps, pt.distance(LineString(points[i:i + 2]))))

            dist = np.array(dist)**2
            vx2 = np.sum(np.array(nlx) / dist) / np.sum(1.0 / np.array(dist))
            vy2 = np.sum(np.array(nly) / dist) / np.sum(1.0 / np.array(dist))

            v2mod = np.sqrt(vx2**2 + vy2**2)
            vx2 = vx2 / v2mod
            vy2 = vy2 / v2mod

            dist_pl = np.exp(-pt.distance(LineString(points[0:]))**2 /
                             (2.0 * d1**2))
            vx[j_bottom + idy,
               i_left + idx] = dist_pl * vx2 + (1.0 - dist_pl) * vx1
            vy[j_bottom + idy,
               i_left + idx] = dist_pl * vy2 + (1.0 - dist_pl) * vy1

            vmod = np.sqrt(vx[j_bottom + idy, i_left + idx]**2 +
                           vy[j_bottom + idy, i_left + idx]**2)

            if (vmod > 0):
                vx[j_bottom + idy,
                   i_left + idx] = vx[j_bottom + idy, i_left + idx] / vmod
                vy[j_bottom + idy,
                   i_left + idx] = vy[j_bottom + idy, i_left + idx] / vmod

            dist_pl = np.exp(-pt.distance(LineString(points[0:-1]))**2 /
                             (2.0 * d2**2))

            distxy[j_bottom + idy, i_left + idx] = dist_pl

    print('Channel map completed')
    print('')

try:

    from input_data_advanced import restart_files
    from input_data_advanced import restart_filling_parameters
    print('Restart files', restart_files)
    n_restarts = len(restart_files)

except ImportError:

    print('')
    print('Restart_files not used')
    n_restarts = 0

# load restart files (if existing)
for i_restart in range(0, n_restarts):

    print("Read restart file ", restart_files[i_restart])
    Zflow_old = np.zeros((ny, nx))

    source = restart_files[i_restart]
    file_exists = exists(source)
    if not file_exists:
        print(source + ' not found.')
        quit()

    hdr = [getline(source, i) for i in range(1, 7)]

    try:
        values_restart = [float(h.split(" ")[-1].strip()) for h in hdr]
    except ValueError:
        print("An problem occurred with header of file ", source)
        print(hdr)

    cols_re, rows_re, lx_re, ly_re, cell_re, nd_re = values_restart

    if values[0:5] != values_restart[0:5]:

        print('Check on restart size failed')

        print(values[0:5])
        print(values_restart[0:5])
        sys.exit(0)

    else:

        print('Check on restart size OK')

    # Load the previous flow thickness into a numpy array
    if crop_flag:

        specific_rows = list(np.arange(6+rows_re-jN)) + \
            list(np.arange(6+rows_re-jS, 6+rows_re))
        specific_columns = list(np.arange(iW, iE))
        arr_df = pd.read_csv(source,
                             delimiter=' ',
                             header=None,
                             usecols=specific_columns,
                             skiprows=specific_rows,
                             skipinitialspace=True)
        arr = arr_df.to_numpy()
        arr[arr == nd_re] = 0.0
        arr = np.flipud(arr)

    else:

        arr_df = pd.read_csv(source,
                             delimiter=' ',
                             skiprows=6,
                             header=None,
                             skipinitialspace=True)
        arr = arr_df.to_numpy()
        arr[arr == nd_re] = 0.0
        arr = np.flipud(arr)

    Zflow_old = arr

    # print(np.where(Zflow_old==np.amax(Zflow_old)))

    # Load the relevant filling_parameter (to account for "subsurface flows")
    filling_parameter_i = restart_filling_parameters[i_restart]

    Zc = Zc + (Zflow_old * filling_parameter_i)
    print("Restart file read")

# Define a small grid for lobe-cells intersection
nv = 15
xv, yv = np.meshgrid(np.linspace(-0.5 * cell, 0.5 * cell, nv),
                     np.linspace(-0.5 * cell, 0.5 * cell, nv))
xv = np.reshape(xv, -1)
yv = np.reshape(yv, -1)
nv2 = nv * nv

Ztot = np.zeros((ny, nx))

# the first argument is the destination, the second is the source
np.copyto(Ztot, Zc)

Zflow = np.zeros((ny, nx))

max_semiaxis = np.sqrt(lobe_area * max_aspect_ratio / np.pi)
max_cells = np.ceil(2.0 * max_semiaxis / cell) + 2
max_cells = max_cells.astype(int)

print('max_semiaxis', max_semiaxis)
print('max_cells', max_cells)

jtop_array = np.zeros(alloc_n_lobes, dtype=int)
jbottom_array = np.zeros(alloc_n_lobes, dtype=int)

iright_array = np.zeros(alloc_n_lobes, dtype=int)
ileft_array = np.zeros(alloc_n_lobes, dtype=int)

Zhazard = np.zeros((ny, nx), dtype=int)
Zhazard_temp = np.zeros((ny, nx), dtype=int)

Zdist = np.zeros((ny, nx), dtype=int) + 9999

print('End pre-processing')
print('')

if sys.version_info >= (3, 0):
    start = time.process_time()
else:
    start = time.clock()

est_rem_time = ''

n_lobes_tot = 0

for flow in range(0, n_flows):

    Zflow_local_array = np.zeros((alloc_n_lobes, max_cells, max_cells),
                                 dtype=int)
    descendents = np.zeros(alloc_n_lobes, dtype=int)

    i_first_check = n_check_loop

    if (a_beta == 0) and (b_beta == 0):

        # DEFINE THE NUMBER OF LOBES OF THE FLOW (RANDOM VALUE BETWEEN
        # MIN AND MAX)
        n_lobes = int(
            np.ceil(np.random.uniform(min_n_lobes, max_n_lobes, size=1)))

    else:

        x_beta = (1.0 * flow) / (n_flows - 1)
        n_lobes = int(
            np.rint(min_n_lobes + 0.5 * (max_n_lobes - min_n_lobes) *
                    beta.pdf(x_beta, a_beta, b_beta)))

    n_lobes_tot = n_lobes_tot + n_lobes

    thickness_min = 2.0 * thickness_ratio / \
        (thickness_ratio + 1.0) * avg_lobe_thickness
    delta_lobe_thickness = 2.0 * \
        (avg_lobe_thickness - thickness_min) / (n_lobes - 1.0)

    # print ('n_lobes',n_lobes)
    # print ('thickness_min',thickness_min)
    # print ('delta_lobe_thickness',delta_lobe_thickness)

    if (n_flows > 1 and not ('SLURM_JOB_NAME' in os.environ.keys())):
        # print on screen bar with percentage of flows computed
        last_percentage_5 = np.rint(flow * 20.0 / (n_flows)).astype(int)
        last_percentage = np.rint(flow * 100.0 / (n_flows))
        last_percentage = np.rint(flow * 100.0 / (n_flows))
        last_percentage = last_percentage.astype(int)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%% %s" %
                         ('=' *
                          (last_percentage_5), last_percentage, est_rem_time))
        sys.stdout.flush()

    for i in range(0, n_init):

        if (n_flows == 1 and not ('SLURM_JOB_NAME' in os.environ.keys())):
            # print on screen bar with percentage of flows computed
            last_percentage = np.rint(i * 20.0 / (n_lobes - 1)) * 5
            last_percentage = last_percentage.astype(int)

            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" %
                             ('=' * (last_percentage / 5), last_percentage))
            sys.stdout.flush()
        else:
            pass

        # STEP 0: COMPUTE THE FIRST LOBES OF EACH FLOW

        if (n_vents == 1):

            x[i] = x_vent[0]
            y[i] = y_vent[0]

        else:

            if (vent_flag == 0):

                # vent_flag = 0  => the initial lobes are on the vents
                #                   coordinates and the flows start initially
                #                   from the first vent, then from the second
                #                   and so on.

                i_vent = int(np.floor(flow * n_vents / n_flows))

                x[i] = x_vent[i_vent]
                y[i] = y_vent[i_vent]

            elif (vent_flag == 1):

                # vent_flag = 1  => the initial lobes are chosen randomly from
                #                   the vents coordinates and each vent has the
                #                   same probability

                i_vent = np.random.randint(n_vents, size=1)

                x[i] = x_vent[int(i_vent)]
                y[i] = y_vent[int(i_vent)]

            elif ((vent_flag == 2) or (vent_flag == 6)):

                # vent_flag = 2  => the initial lobes are on the polyline
                #                   connecting the vents and all the point of
                #                   the polyline have the same probability

                # vent_flag = 6  => the initial lobes are on the polyline
                #                   connecting the vents and the probability of
                #                   each segment is fixed in the input file

                alfa_polyline = np.random.uniform(0, 1, size=1)

                idx_vent = np.argmax(cum_fiss_length > alfa_polyline)

                num = alfa_polyline - cum_fiss_length[idx_vent - 1]
                den = cum_fiss_length[idx_vent] - cum_fiss_length[idx_vent - 1]

                alfa_segment = num / den

                x[i] = alfa_segment * x_vent[idx_vent] + \
                    (1.0 - alfa_segment) * x_vent[idx_vent-1]

                y[i] = alfa_segment * y_vent[idx_vent] + \
                    (1.0 - alfa_segment) * y_vent[idx_vent-1]

            elif (vent_flag == 3):

                # vent_flag = 3  => the initial lobes are on the polyline
                #                   connecting the vents and all the segments
                #                   of the polyline have the same probability

                i_segment = randrange(n_vents)

                alfa_segment = np.random.uniform(0, 1, size=1)

                x[i] = alfa_segment * x_vent[i_segment] + \
                    (1.0 - alfa_segment) * x_vent[i_segment-1]

                y[i] = alfa_segment * y_vent[i_segment] + \
                    (1.0 - alfa_segment) * y_vent[i_segment-1]

            elif ((vent_flag == 4) or (vent_flag == 7)):

                # vent_flag = 4  => the initial lobes are on multiple
                #                   fissures and all the point of the fissures
                #                   have the same probability

                # vent_flag = 7  => the initial lobes are on multiple
                #                   fissures and the probability of
                #                   each fissure is fixed in the input file

                alfa_polyline = np.random.uniform(0, 1, size=1)

                idx_vent = np.argmax(cum_fiss_length > alfa_polyline)

                num = alfa_polyline - cum_fiss_length[idx_vent - 1]
                den = cum_fiss_length[idx_vent] - cum_fiss_length[idx_vent - 1]

                alfa_segment = num / den
                print()
                print(idx_vent - 1, alfa_segment)

                x[i] = alfa_segment * x_vent_end[idx_vent-1] + \
                    (1.0 - alfa_segment) * x_vent[idx_vent-1]

                y[i] = alfa_segment * y_vent_end[idx_vent-1] + \
                    (1.0 - alfa_segment) * y_vent[idx_vent-1]

            elif (vent_flag == 5):

                # vent_flag = 5  => the initial lobes are on multiple
                #                   fissures and all the fissures
                #                   have the same probability

                i_segment = randrange(n_vents)

                alfa_segment = np.random.uniform(0, 1, size=1)

                x[i] = alfa_segment * x_vent_end[i_segment] + \
                    (1.0 - alfa_segment) * x_vent[i_segment]

                y[i] = alfa_segment * y_vent_end[i_segment] + \
                    (1.0 - alfa_segment) * y_vent[i_segment]

            elif (vent_flag == 8):

                # vent_flag = 1  => the initial lobes are chosen randomly from
                #                   the vents coordinates and each vent has
                #                   the same probability

                alfa_vent = np.random.uniform(0, 1, size=1)
                i_vent = np.argmax(cum_fiss_length > alfa_vent)

                x[i] = x_vent[int(i_vent)]
                y[i] = y_vent[int(i_vent)]

        # initialize distance from first lobe and number of descendents
        dist_int[i] = 0
        descendents[i] = 0

        # compute the gradient of the topography(+ eventually the flow)
        # here the centered grid is used (Z values saved at the centers of
        # the pixels)
        # xc[ix] < lobe_center_x < xc[ix1]
        # yc[iy] < lobe_center_y < yc[iy1]
        xi = (x[i] - xcmin) / cell
        yi = (y[i] - ycmin) / cell

        ix = np.floor(xi)
        iy = np.floor(yi)

        ix = ix.astype(int)
        iy = iy.astype(int)

        # compute the baricentric coordinated of the lobe center in the pixel
        # 0 < xi_fract < 1
        # 0 < yi_fract < 1
        xi_fract = xi - ix
        yi_fract = yi - iy

        # interpolate the slopes at the edges of the pixel to find the slope
        # at the center of the lobe
        Fx_test = (yi_fract * (Ztot[iy + 1, ix + 1] - Ztot[iy + 1, ix]) +
                   (1.0 - yi_fract) * (Ztot[iy, ix + 1] - Ztot[iy, ix])) / cell

        Fy_test = (xi_fract * (Ztot[iy + 1, ix + 1] - Ztot[iy, ix + 1]) +
                   (1.0 - xi_fract) * (Ztot[iy + 1, ix] - Ztot[iy, ix])) / cell

        # major semi-axis direction
        max_slope_angle = np.mod(
            180.0 + (180.0 * np.arctan2(Fy_test, Fx_test) / np.pi), 360.0)

        # slope of the topography at (x[0],y[0])
        slope = np.sqrt(np.square(Fx_test) + np.square(Fy_test))

        # PERTURBE THE MAXIMUM SLOPE ANGLE ACCORDING TO PROBABILITY LAW

        # this expression define a coefficient used for the direction of the
        # next slope
        if (max_slope_prob < 1):

            # angle defining the direction of the new slope. when slope=0, then
            # we have an uniform distribution for the possible angles for the
            # next lobe.

            slopedeg = 180.0 * np.arctan(slope) / np.pi

            if (slopedeg > 0.0) and (max_slope_prob > 0):

                sigma = (1.0 - max_slope_prob) / max_slope_prob * \
                    (90.0 - slopedeg) / slopedeg
                rand_angle_new = rtnorm.rtnorm(-180, 180, 0, sigma)

            else:

                rand = np.random.uniform(0, 1, size=1)
                rand_angle_new = 360.0 * np.abs(rand - 0.5)

            angle[i] = max_slope_angle + rand_angle_new

        else:

            angle[i] = max_slope_angle

        # factor for the lobe eccentricity
        aspect_ratio = min(max_aspect_ratio, 1.0 + aspect_ratio_coeff * slope)

        # semi-axes of the lobe:
        # x1(i) is the major semi-axis of the lobe;
        # x2(i) is the minor semi-axis of the lobe.
        x1[i] = np.sqrt(lobe_area / np.pi) * np.sqrt(aspect_ratio)
        x2[i] = np.sqrt(lobe_area / np.pi) / np.sqrt(aspect_ratio)

        if (saveraster_flag == 1):

            # compute the points of the lobe
            [xe, ye] = ellipse(x[i], y[i], x1[i], x2[i], angle[i], X_circle,
                               Y_circle)

            # bounding box for the lobe
            # (xc[i_left]<xe<xc[i_right];yc[j_bottom]<ye<yc[j_top])
            # the indexes are referred to the centers of the pixels
            min_xe = np.min(xe)
            max_xe = np.max(xe)

            min_ye = np.min(ye)
            max_ye = np.max(ye)

            xi = (min_xe - xcmin) / cell
            ix = np.floor(xi)
            i_left = ix.astype(int)

            xi = (max_xe - xcmin) / cell
            ix = np.floor(xi)
            i_right = ix.astype(int) + 2

            yj = (min_ye - ycmin) / cell
            jy = np.floor(yj)
            j_bottom = jy.astype(int)

            yj = (max_ye - ycmin) / cell
            jy = np.floor(yj)
            j_top = jy.astype(int) + 2

            # define the subgrid of pixels to check for coverage
            Xc_local = Xc[j_bottom:j_top, i_left:i_right]
            Yc_local = Yc[j_bottom:j_top, i_left:i_right]

            # compute the fraction of cells covered by the lobe (local index)
            # for each pixel a square [-0.5*cell;0.5*cell]X[-0.5*cell;0.5*cell]
            # is built around its center to compute the intersection with the
            # lobe the coverage values are associated to each pixel (at the
            # center)
            area_fract = local_intersection(Xc_local, Yc_local, x[i], y[i],
                                            x1[i], x2[i], angle[i], xv, yv,
                                            nv2)
            Zflow_local = area_fract

            # compute also as integer (0=pixel non covereb by lobe;1=pixel
            # covered by lobe)
            Zflow_local_int = np.ceil(area_fract)
            Zflow_local_int = Zflow_local_int.astype(int)

            # compute the thickness of the lobe
            lobe_thickness = thickness_min + (i - 1) * delta_lobe_thickness

            # update the thickness of the flow with the new lobe
            Zflow[j_bottom:j_top,
                  i_left:i_right] += lobe_thickness * Zflow_local

            # update the topography

            # change 2022/01/13
            # FROM HERE
            Ztot[j_bottom:j_top, i_left:i_right] += filling_parameter[
                j_bottom:j_top, i_left:i_right] * lobe_thickness * Zflow_local

            # TO HERE

            # compute the new minimum "lobe distance" of the pixels from the
            # vent
            Zdist_local = Zflow_local_int * \
                dist_int[i] + 9999 * (Zflow_local == 0)

            Zdist[j_bottom:j_top, i_left:i_right] = np.minimum(
                Zdist[j_bottom:j_top, i_left:i_right], Zdist_local)

            # store the bounding box of the new lobe
            jtop_array[i] = j_top
            jbottom_array[i] = j_bottom

            iright_array[i] = i_right
            ileft_array[i] = i_left

            if (hazard_flag):

                # store the local array of integer coverage in the global array
                Zflow_local_array[i, 0:j_top - j_bottom,
                                  0:i_right - i_left] = Zflow_local_int

    last_lobe = n_lobes

    for i in range(n_init, n_lobes):

        # print('i',i)

        if (n_flows == 1 and 'SLURM_JOB_NAME' not in os.environ.keys()):
            # print on screen bar with percentage of flows computed
            last_percentage = np.rint(i * 20.0 / (n_lobes - 1)) * 5
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" %
                             ('=' * (last_percentage / 5), last_percentage))
            sys.stdout.flush()

        # STEP 0: DEFINE THE INDEX idx OF THE PARENT LOBE

        if (lobe_exponent > 0):

            idx0 = np.random.uniform(0, 1, size=1)

            idx1 = idx0**lobe_exponent

            if (force_max_length):

                # the parent lobe is chosen only among those with
                # dist smaller than the maximum value fixed in input
                mask = dist_int[0:i] < max_length

                idx2 = sum(mask[0:i]) * idx1

                idx3 = np.floor(idx2)

                idx = int(idx3)

                sorted_dist = np.argsort(dist_int[0:i])

                idx = sorted_dist[idx]

            else:

                # the parent lobe is chosen among all the lobes

                idx2 = i * idx1

                idx3 = np.floor(idx2)

                idx = int(idx3)

            if (start_from_dist_flag):

                # the probability law is associated to the distance
                # from the vent
                sorted_dist = np.argsort(dist_int[0:i])

                idx = sorted_dist[idx]

        else:

            idx = i - 1

        # save the index of the parent and the distance from first lobe of the
        # chain
        parent[i] = idx
        dist_int[i] = dist_int[idx] + 1

        # for all the "ancestors" increase by one the number of descendents

        last = i

        for j in range(0, dist_int[idx] + 1):

            previous = parent[last]
            descendents[previous] = descendents[previous] + 1
            last = previous

        # local slope of the topography. The slope affects both the location of
        # the new lobe on the boundary of the previous one and its aspect
        # ratio:
        # if slope = 0 the lobe is a circle (x1=x2);
        # if slope > 1 the lobe is an ellipse.

        # STEP 1: COMPUTE THE SLOPE AND THE MAXIMUM SLOPE ANGLE
        # here the centered grid is used (Z values saved at the centers of the
        # pixels)
        # xc[ix] < lobe_center_x < xc[ix1]
        # yc[iy] < lobe_center_y < yc[iy1]
        xi = (x[idx] - xcmin) / cell
        yi = (y[idx] - ycmin) / cell

        ix = np.floor(xi)
        iy = np.floor(yi)

        ix = ix.astype(int)
        iy = iy.astype(int)

        ix1 = ix + 1
        iy1 = iy + 1

        # stopping condition (lobe close the domain boundary)
        if (ix <= 0.5 * max_cells) or (ix1 >= (nx - 0.5*max_cells)) or \
           (iy <= 0.5 * max_cells) or (iy1 >= (ny - 0.5*max_cells)) or \
           (Zc[iy, ix] == nd) or (Zc[iy1, ix1] == nd) or \
           (Zc[iy, ix1] == nd) or (Zc[iy1, ix] == nd):

            last_lobe = i - 1
            break

        # compute the baricentric coordinated of the lobe center in the pixel
        # 0 < xi_fract < 1
        # 0 < yi_fract < 1
        xi_fract = xi - ix
        yi_fract = yi - iy

        # interpolate the elevation at the corners of the pixel to find the
        # elevation at the center of the lobe
        zidx = xi_fract * (yi_fract * Ztot[iy1, ix1] +
                           (1.0 - yi_fract) * Ztot[iy, ix1]) + (
                               1.0 -
                               xi_fract) * (yi_fract * Ztot[iy1, ix] +
                                            (1.0 - yi_fract) * Ztot[iy, ix])
        """
        # interpolate the slopes at the edges of the pixel to find the slope
        # at the center of the lobe
        Fx_lobe = ( yi_fract * ( Ztot[iy1,ix1] - Ztot[iy1,ix] ) \
                    + (1.0-yi_fract) * ( Ztot[iy,ix1] - Ztot[iy,ix] ) ) / cell

        Fy_lobe = ( xi_fract * ( Ztot[iy1,ix1] - Ztot[iy,ix1] ) \
                    + (1.0-xi_fract) * ( Ztot[iy1,ix] - Ztot[iy,ix] ) ) / cell


        slope = np.sqrt(np.square(Fx_lobe)+np.square(Fy_lobe))
        # angle defining the direction of maximum slope
        # (max_slope_angle = aspect)
        max_slope_angle = np.mod(
            180 + ( 180 * np.arctan2(Fy_lobe,Fx_lobe) / np.pi ),360.0)
        """

        # compute the lobe (npoints on the ellipse)
        [xe, ye] = ellipse(x[idx], y[idx], x1[idx], x2[idx], angle[idx],
                           X_circle, Y_circle)

        # For all the points of the ellipse compute the indexes of the pixel
        # containing the points. This is done with respect to the centered
        # grid. We want to interpolate from the centered values (elevation)
        # to the location of the points on the ellipse)
        xei = (xe - xcmin) / cell
        yei = (ye - ycmin) / cell

        ixe = np.floor(xei)
        iye = np.floor(yei)

        ixe = ixe.astype(int)
        iye = iye.astype(int)

        ixe1 = ixe + 1
        iye1 = iye + 1

        # compute the local coordinates of the points (0<x,y<1) within the
        # pixels containing them
        xei_fract = xei - ixe
        yei_fract = yei - iye

        # interpolate the grid values to find the elevation at the ellipse
        # points
        ze = xei_fract * (yei_fract * Ztot[iye1, ixe1] +
                          (1.0 - yei_fract) * Ztot[iye, ixe1]) + (
                              1.0 -
                              xei_fract) * (yei_fract * Ztot[iye1, ixe] +
                                            (1.0 - yei_fract) * Ztot[iye, ixe])

        # find the point on the ellipse with minimum elevation
        idx_min = np.argmin(ze)

        # compute the vector from the center of the lobe to the point of
        # minimum z on the boundary
        Fx_lobe = x[idx] - xe[idx_min]
        Fy_lobe = y[idx] - ye[idx_min]

        # compute the slope and the angle
        slope = np.maximum(0.0, (zidx - ze[idx_min]) /
                           (np.sqrt(np.square(Fx_lobe) + np.square(Fy_lobe))))

        max_slope_angle = np.mod(
            180.0 + (180.0 * np.arctan2(Fy_lobe, Fx_lobe) / np.pi), 360.0)

        # STEP 2: PERTURBE THE MAXIMUM SLOPE ANGLE ACCORDING TO PROBABILITY LAW

        # this expression define a coefficient used for the direction of the
        # next slope
        if (max_slope_prob < 1):

            # angle defining the direction of the new slope. when slope=0, then
            # we have an uniform distribution for the possible angles for the
            # next lobe.

            slopedeg = 180.0 * np.arctan(slope) / np.pi

            if (slopedeg > 0.0) and (max_slope_prob > 0.0):

                sigma = (1.0 - max_slope_prob) / max_slope_prob * (
                    90.0 - slopedeg) / slopedeg
                rand_angle_new = rtnorm.rtnorm(-180.0, 180.0, 0.0, sigma)

            else:

                rand = np.random.uniform(0, 1, size=1)
                rand_angle_new = 360.0 * np.abs(rand - 0.5)

            new_angle = max_slope_angle + rand_angle_new[0]

        else:

            new_angle = max_slope_angle

        # STEP 3: ADD THE EFFECT OF INERTIA

        # cos and sin of the angle of the parent lobe
        cos_angle1 = np.cos(angle[idx] * deg2rad)
        sin_angle1 = np.sin(angle[idx] * deg2rad)

        # cos and sin of the angle of maximum slope
        cos_angle2 = np.cos(new_angle * deg2rad)
        sin_angle2 = np.sin(new_angle * deg2rad)

        if (inertial_exponent == 0):

            alfa_inertial[i] = 0.0

        else:

            alfa_inertial[i] = (1.0 - (2.0 * np.arctan(slope) / np.pi) **
                                inertial_exponent)**(1.0 / inertial_exponent)

        x_avg = (1.0 - alfa_inertial[i]) * \
            cos_angle2 + alfa_inertial[i] * cos_angle1
        y_avg = (1.0 - alfa_inertial[i]) * \
            sin_angle2 + alfa_inertial[i] * sin_angle1

        angle_avg = np.mod(180 * np.arctan2(y_avg, x_avg) / np.pi, 360)

        new_angle = angle_avg

        if (alfa_channel > 0.0):

            old_angle = new_angle

            # interpolate the vector at the corners of the pixel to find the
            # vector at the center of the lobe
            cos_angle_old = np.cos(np.radians(old_angle))
            sin_angle_old = np.sin(np.radians(old_angle))

            # print('cos_angle1,sin_angle1',cos_angle1,sin_angle1)

            x_avg2 = xi_fract * (
                yi_fract * vx[iy1, ix1] + (1.0 - yi_fract) * vx[iy, ix1]) + (
                    1.0 - xi_fract) * (yi_fract * vx[iy1, ix] +
                                       (1.0 - yi_fract) * vx[iy, ix])
            y_avg2 = xi_fract * (
                yi_fract * vy[iy1, ix1] + (1.0 - yi_fract) * vy[iy, ix1]) + (
                    1.0 - xi_fract) * (yi_fract * vy[iy1, ix] +
                                       (1.0 - yi_fract) * vy[iy, ix])

            if (x_avg2**2 + y_avg2**2 > 0.0):

                cos_angle_new = x_avg2 / np.sqrt(x_avg2**2 + y_avg2**2)
                sin_angle_new = y_avg2 / np.sqrt(x_avg2**2 + y_avg2**2)

                # print('cos_angle2,sin_angle2',cos_angle2,sin_angle2)

                distxyidx = xi_fract * (
                    yi_fract * distxy[iy1, ix1] +
                    (1.0 - yi_fract) * distxy[iy, ix1]) + (
                        1.0 - xi_fract) * (yi_fract * distxy[iy1, ix] +
                                           (1.0 - yi_fract) * distxy[iy, ix])

                x_avg = (1.0 - alfa_channel*distxyidx) * \
                    cos_angle_old + alfa_channel*distxyidx * cos_angle_new
                y_avg = (1.0 - alfa_channel*distxyidx) * \
                    sin_angle_old + alfa_channel*distxyidx * sin_angle_new

                angle_avg = np.mod(180.0 * np.arctan2(y_avg, x_avg) / np.pi,
                                   360.0)

                new_angle = angle_avg

        # STEP 4: DEFINE THE SEMI-AXIS OF THE NEW LOBE

        # a define the ang.coeff. of the line defining the location of the
        # center of the new lobe in a coordinate system defined by the
        # semi-axes of the existing lobe
        a = np.tan(deg2rad * (new_angle - angle[idx]))

        # xt is the 1st-coordinate of the point of the boundary of the ellipse
        # definind the direction of the new lobe, in a coordinate system
        # defined by the semi-axes of the existing lobe
        if (np.cos(deg2rad * (new_angle - angle[idx])) > 0):

            xt = np.sqrt(x1[idx]**2 * x2[idx]**2 /
                         (x2[idx]**2 + x1[idx]**2 * a**2))

        else:

            xt = -np.sqrt(x1[idx]**2 * x2[idx]**2 /
                          (x2[idx]**2 + x1[idx]**2 * a**2))

        # yt is the 2nd-coordinate of the point of the boundary of the ellipse
        # definind the direction of the new lobe, in a coordinate system
        # defined by the semi-axes of the existing lobe
        yt = a * xt

        # (delta_x,delta_y) is obtained rotating the vector (xt,yt) by the
        # angle defined by the major semi-axis of the existing lobe. In this
        # way we obtain the location in a coordinate-system centered in the
        # center of the existing lobe, but this time with the axes parallel to
        # the original x and y axes.

        delta_x = xt * cos_angle1 - yt * sin_angle1
        delta_y = xt * sin_angle1 + yt * cos_angle1

        # the slope coefficient is evaluated at the point of the boundary of
        # the ellipse definind by the direction of the new lobe

        xi = (x[idx] + delta_x - xcmin) / cell
        yi = (y[idx] + delta_y - ycmin) / cell

        ix = np.floor(xi)
        iy = np.floor(yi)

        ix = ix.astype(int)
        iy = iy.astype(int)

        ix1 = ix + 1
        iy1 = iy + 1

        # stopping condition (lobe close the domain boundary)
        if (ix <= 0.5 * max_cells) or (ix1 >= nx - 0.5*max_cells) or \
           (iy <= 0.5 * max_cells) or (iy1 >= ny - 0.5*max_cells):

            # print('ix',ix,'iy',iy)
            last_lobe = i - 1
            break

        xi_fract = xi - ix
        yi_fract = yi - iy

        # ztot at the new budding point
        ze = xi_fract * (yi_fract * Ztot[iy1, ix1] +
                         (1.0 - yi_fract) * Ztot[iy, ix1]) + (
                             1.0 -
                             xi_fract) * (yi_fract * Ztot[iy1, ix] +
                                          (1.0 - yi_fract) * Ztot[iy, ix])

        slope = np.maximum(0.0, (zidx - ze) /
                           (np.sqrt(np.square(delta_x) + np.square(delta_y))))

        aspect_ratio = min(max_aspect_ratio, 1.0 + aspect_ratio_coeff * slope)

        # (new_x1,new_x2) are the semi-axes of the new lobe. slope_coeff is
        # used to have an elongated lobe accoriding to the slope of the
        # topography. It is possible to modifiy these values in order to have
        # the same volume for all the lobes.
        new_x1 = np.sqrt(lobe_area / np.pi) * np.sqrt(aspect_ratio)
        new_x2 = np.sqrt(lobe_area / np.pi) / np.sqrt(aspect_ratio)

        # v1 is the distance of the new point found on the boundary of the lobe
        # from the center of the lobe
        v1 = np.sqrt(delta_x**2 + delta_y**2)

        # v2 is the distance between the centers of the two lobes when they
        # intersect in one point only
        v2 = v1 + new_x1

        # v is the distance between the centers of the two lobes, according to
        # the value of the parameter dist_fact
        v = (v1 * (1.0 - dist_fact) + v2 * dist_fact) / v1

        # STEP 5: BUILD THE NEW LOBE

        # (x_new,y_new) are the coordinates of the center of the new lobe
        x_new = x[idx] + v * delta_x
        y_new = y[idx] + v * delta_y

        # store the parameters of the new lobe in arrays
        angle[i] = new_angle
        x1[i] = new_x1
        x2[i] = new_x2
        x[i] = x_new
        y[i] = y_new

        # check the grid points covered by the lobe
        if (saveraster_flag == 1):

            # compute the new lobe
            [xe, ye] = ellipse(x[i], y[i], x1[i], x2[i], angle[i], X_circle,
                               Y_circle)

            # bounding box for the new lobe
            # the indexes are referred to the centers of the pixels
            min_xe = np.min(xe)
            max_xe = np.max(xe)

            min_ye = np.min(ye)
            max_ye = np.max(ye)

            xi = (min_xe - xcmin) / cell
            ix = np.floor(xi)
            i_left = ix.astype(int)
            i_left = np.maximum(0, np.minimum(nx - 1, i_left))

            xi = (max_xe - xcmin) / cell
            ix = np.floor(xi)
            i_right = ix.astype(int) + 2
            i_right = np.maximum(0, np.minimum(nx - 1, i_right))

            yj = (min_ye - ycmin) / cell
            jy = np.floor(yj)
            j_bottom = jy.astype(int)
            j_bottom = np.maximum(0, np.minimum(ny - 1, j_bottom))

            yj = (max_ye - ycmin) / cell
            jy = np.floor(yj)
            j_top = jy.astype(int) + 2
            j_top = np.maximum(0, np.minimum(ny - 1, j_top))

            # the centers of the pixels are used to compute the intersection
            # with the lobe
            Xc_local = Xc[j_bottom:j_top, i_left:i_right]
            Yc_local = Yc[j_bottom:j_top, i_left:i_right]

            # compute the fraction of cells covered by the lobe (local index)
            # for each pixel a square [-0.5*cell;0.5*cell]X[-0.5*cell;0.5*cell]
            # is built around its center to compute the intersection with the
            # lobe the coverage values are associated to each pixel (at the
            # center)
            area_fract = local_intersection(Xc_local, Yc_local, x[i], y[i],
                                            x1[i], x2[i], angle[i], xv, yv,
                                            nv2)

            Zflow_local = area_fract

            # compute the local integer covering (0-not covered  1-covered)
            Zflow_local_int = np.ceil(area_fract)
            Zflow_local_int = Zflow_local_int.astype(int)

            # print('Zflow_local_int')
            # print(Zflow_local_int)

            # define the distance (number of lobes) from the vent (local index)
            Zdist_local = Zflow_local_int * \
                dist_int[i] + 9999 * (Zflow_local == 0)

            # update the minimum distance in the global indexing
            Zdist[j_bottom:j_top, i_left:i_right] = np.minimum(
                Zdist[j_bottom:j_top, i_left:i_right], Zdist_local)

            # compute the thickness of the lobe
            lobe_thickness = thickness_min + (i - 1) * delta_lobe_thickness

            # update the thickness for the grid points selected (global index)
            Zflow[j_bottom:j_top,
                  i_left:i_right] += lobe_thickness * Zflow_local

            # change 2022/01/13

            Ztot[j_bottom:j_top, i_left:i_right] += filling_parameter[
                j_bottom:j_top, i_left:i_right] * lobe_thickness * Zflow_local
            # TO HERE

            # save the bounding box of the i-th lobe
            jtop_array[i] = j_top
            jbottom_array[i] = j_bottom

            iright_array[i] = i_right
            ileft_array[i] = i_left

            if (hazard_flag):

                # store the local arrays used later for the hazard map

                if not (Zflow_local_int.shape[0] == (j_top - j_bottom)):

                    print(Zflow_local_int.shape[0], j_top, j_bottom)
                    print(Zflow_local_int.shape[1], i_right, i_left)
                    print('')

                if not (Zflow_local_int.shape[1] == (i_right - i_left)):

                    print(Zflow_local_int.shape[0], j_top, j_bottom)
                    print(Zflow_local_int.shape[1], i_right, i_left)
                    print('')

                if (np.max(Zflow_local.shape) > Zflow_local_array.shape[1]):

                    print('check 3')
                    print(cell, new_x1, new_x2, new_angle)
                    print(x[i], y[i], x1[i], x2[i])
                    np.set_printoptions(precision=1)
                    print(Zflow_local_int)

                Zflow_local_array[i, 0:j_top - j_bottom,
                                  0:i_right - i_left] = Zflow_local_int

    if (hazard_flag):

        # update the hazard map accounting for the number of descendents,
        # representative of the number of times a flow has passed over a cell

        for i in range(0, last_lobe):

            j_top = jtop_array[i]
            j_bottom = jbottom_array[i]

            i_right = iright_array[i]
            i_left = ileft_array[i]

            if (i > 0):

                j_top_int = np.minimum(j_top, jtop_array[parent[i]])
                j_bottom_int = np.maximum(j_bottom, jbottom_array[parent[i]])
                i_left_int = np.maximum(i_left, ileft_array[parent[i]])
                i_right_int = np.minimum(i_right, iright_array[parent[i]])

                Zlocal_new = np.zeros((max_cells, max_cells), dtype=int)
                Zlocal_parent = np.zeros((max_cells, max_cells), dtype=int)

                Zlocal_parent = Zflow_local_array[
                    parent[i],
                    np.maximum(0, j_bottom_int - jbottom_array[parent[i]]):np.
                    minimum(j_top_int -
                            jbottom_array[parent[i]], jtop_array[parent[i]] -
                            jbottom_array[parent[i]]),
                    np.maximum(i_left_int - ileft_array[parent[i]], 0):np.
                    minimum(i_right_int -
                            ileft_array[parent[i]], iright_array[parent[i]] -
                            ileft_array[parent[i]])]

                Zlocal_new = Zflow_local_array[i, 0:j_top - j_bottom,
                                               0:i_right - i_left]

                if (Zlocal_parent.shape[0] == 0
                        or Zlocal_parent.shape[1] == 0):

                    print('check')
                    print('idx', i)
                    print('j', j_bottom, j_top)
                    print('i', i_left, i_right)
                    print('idx parent', parent[i])
                    print('j', jbottom_array[parent[i]], jtop_array[parent[i]])
                    print('i', ileft_array[parent[i]], iright_array[parent[i]])
                    print(j_bottom_int, j_top_int, i_left_int, i_right_int)

                Zlocal_new[
                    np.maximum(0, j_bottom_int -
                               j_bottom):np.minimum(j_top_int -
                                                    j_bottom, j_top -
                                                    j_bottom),
                    np.maximum(i_left_int -
                               i_left, 0):np.minimum(i_right_int -
                                                     i_left, i_right -
                                                     i_left)] *= (
                                                         1 - Zlocal_parent)

                Zhazard[j_bottom:j_top, i_left:i_right] += descendents[i] \
                    * Zlocal_new[0:j_top-j_bottom, 0:i_right-i_left]

            else:

                Zhazard[j_bottom:j_top, i_left:i_right] += descendents[i] \
                    * Zflow_local_array[i, 0:j_top-j_bottom, 0:i_right-i_left]

    if sys.version_info >= (3, 0):
        elapsed = (time.process_time() - start)
    else:
        elapsed = (time.clock() - start)

    estimated = np.ceil(elapsed * n_flows / (flow + 1) - elapsed)
    est_rem_time = str(datetime.timedelta(seconds=estimated))

if (n_flows > 1 and 'SLURM_JOB_NAME' not in os.environ.keys()):
    # print on screen bar with percentage of flows computed
    last_percentage = 100
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * 20, last_percentage))
    sys.stdout.flush()

if sys.version_info >= (3, 0):
    elapsed = (time.process_time() - start)
else:
    elapsed = (time.clock() - start)

print('')
print('')
print('Total number of lobes', n_lobes_tot, 'Average number of lobes',
      int(1.0 * n_lobes_tot / n_flows))
print('')
print('Time elapsed ' + str(elapsed) + ' sec.' + ' / ' +
      str(int(elapsed / 60)) + ' min.')
print('')
print('Saving files')

if (saveraster_flag == 1):
    # Save raster files

    header = "ncols     %s\n" % Zflow.shape[1]
    header += "nrows    %s\n" % Zflow.shape[0]
    header += "xllcorner " + str(lx) + "\n"
    header += "yllcorner " + str(ly) + "\n"
    header += "cellsize " + str(cell) + "\n"
    header += "NODATA_value 0\n"

    output_full = run_name + '_thickness_full.asc'

    np.savetxt(output_full,
               np.flipud(Zflow),
               header=header,
               fmt='%1.5f',
               comments='')

    print('')
    print(output_full + ' saved')

    flag_union_diff = False

    try:

        from input_data import union_diff_file

        # Parse the header using a loop and
        # the built-in linecache module
        hdr = [getline(union_diff_file, i) for i in range(1, 7)]
        values = [float(h.split(" ")[-1].strip()) for h in hdr]
        Zs_temp = np.flipud(np.loadtxt(union_diff_file, skiprows=6))

        cols_ud, rows_ud, lx_ud, ly_ud, cell_ud, nd_ud = values
        if (cols_ud != Zflow.shape[1]) or (rows_ud != Zflow.shape[0]) or (
                lx_ud != lx) or (ly_ud != ly) or (cell_ud != cell):

            print("Union_diff_file", union_diff_file)
            print("Different header: interpolating data")

            xin = lx_ud + cell_ud * np.arange(cols_ud)
            yin = ly_ud + cell_ud * np.arange(rows_ud)

            xout = lx + cell_ud * np.arange(nx)
            yout = ly + cell_ud * np.arange(ny)

            Xout, Yout = np.meshgrid(xout, yout)

            Zs1 = interp2Dgrids(xin, yin, Zs_temp, Xout, Yout)

            flag_union_diff = True

        else:

            flag_union_diff = True
            Zs1 = Zs_temp

        Zs2 = Zflow

        Zs_union = np.maximum(Zs1, Zs2)

        Zs_union = Zs_union / np.maximum(Zs_union, 1)
        area_union = np.sum(Zs_union) * cell**2

        Zs_inters = np.minimum(Zs1, Zs2)

        Zs_inters = Zs_inters / np.maximum(Zs_inters, 1)
        area_inters = np.sum(Zs_inters) * cell**2

        # area_inters = np.count_nonzero(Zs_inters) * cell**2
        fitting_parameter = area_inters / area_union

        print('--------------------------------')
        print('With full output')
        print('Union area', area_union, 'Intersect. area', area_inters)
        print('Fitting parameter', fitting_parameter)

        Zs1_mean = np.mean(
            Zs1 * Zs_inters) * nx * ny / np.count_nonzero(Zs_inters)
        Zs2_mean = np.mean(
            Zs2 * Zs_inters) * nx * ny / np.count_nonzero(Zs_inters)

        Zs1_vol = Zs1_mean * area_inters
        Zs2_vol = Zs2_mean * area_inters

        print('Volume 1 in intersection', Zs1_vol, 'Volume 2 in intersection',
              Zs2_vol)

        Zs_diff = np.abs(Zs1 - Zs2)

        Zs_diff = Zs_diff * Zs_inters

        avg_thick_diff = np.mean(Zs_diff) * nx * ny / np.count_nonzero(
            Zs_inters)
        std_thick_diff = np.std(Zs_diff) * nx * ny / np.count_nonzero(
            Zs_inters)
        vol_diff = avg_thick_diff * area_inters

        rel_err_vol = vol_diff / np.maximum(Zs1_vol, Zs2_vol)

        print('Thickness relative error', rel_err_vol)
        print('--------------------------------')

    except ImportError:

        print("Union_diff_file not defined")
        flag_union_diff = False

    if isinstance(masking_threshold, float):

        masking_threshold = [masking_threshold]

    n_masking = len(masking_threshold)

    for i_thr in range(n_masking):

        if (masking_threshold[i_thr] < 1):

            max_lobes = int(np.floor(np.max(Zflow / avg_lobe_thickness)))

            for i in range(1, 10 * max_lobes):

                masked_Zflow = ma.masked_where(
                    Zflow < i * 0.1 * avg_lobe_thickness, Zflow)

                total_Zflow = np.sum(Zflow)

                if (flag_threshold == 1):

                    volume_fraction = np.sum(masked_Zflow) / total_Zflow

                    coverage_fraction = volume_fraction

                else:

                    area_fraction = np.true_divide(np.sum(masked_Zflow > 0),
                                                   np.sum(Zflow > 0))

                    coverage_fraction = area_fraction
                    # print (coverage_fraction)

                if (coverage_fraction < masking_threshold[i_thr]):

                    if (flag_threshold == 1):

                        print('')
                        print('Masking threshold', masking_threshold[i_thr])
                        print('Total volume', cell**2 * total_Zflow,
                              ' m3 Masked volume',
                              cell**2 * np.sum(masked_Zflow),
                              ' m3 Volume fraction', coverage_fraction)
                        print('Total area', cell**2 * np.sum(Zflow > 0),
                              ' m2 Masked area',
                              cell**2 * np.sum(masked_Zflow > 0), ' m2')
                        print('Average thickness full',
                              total_Zflow / np.sum(Zflow > 0),
                              ' m Average thickness mask',
                              np.sum(masked_Zflow) / np.sum(masked_Zflow > 0),
                              ' m')

                    output_thickness = run_name + '_avg_thick.txt'
                    with open(output_thickness, 'a') as the_file:

                        if (i_thr == 0):
                            the_file.write('Average lobe thickness = ' +
                                           str(avg_lobe_thickness) + ' m\n')
                            the_file.write('Total volume = ' +
                                           str(cell**2 * total_Zflow) +
                                           ' m3\n')
                            the_file.write('Total area = ' +
                                           str(cell**2 * np.sum(Zflow > 0)) +
                                           ' m2\n')
                            the_file.write('Average thickness full = ' +
                                           str(total_Zflow /
                                               np.sum(Zflow > 0)) + ' m\n')

                        the_file.write('Masking threshold = ' +
                                       str(masking_threshold[i_thr]) + '\n')
                        the_file.write('Masked volume = ' +
                                       str(cell**2 * np.sum(masked_Zflow)) +
                                       ' m3\n')
                        the_file.write('Masked area = ' +
                                       str(cell**2 *
                                           np.sum(masked_Zflow > 0)) + ' m2\n')
                        the_file.write('Average thickness mask = ' + str(
                            np.sum(masked_Zflow) / np.sum(masked_Zflow > 0)) +
                                       ' m\n')

                    output_masked = run_name + '_thickness_masked' + '_' + \
                        str(masking_threshold[i_thr]).replace(
                            '.', '_') + '.asc'

                    np.savetxt(output_masked,
                               np.flipud((1 - masked_Zflow.mask) * Zflow),
                               header=header,
                               fmt='%1.5f',
                               comments='')

                    print('')
                    print(output_masked + ' saved')

                    break

            if flag_union_diff:

                Zs2 = (1 - masked_Zflow.mask) * Zflow
                Zs_union = np.maximum(Zs1, Zs2)

                Zs_union = Zs_union / np.maximum(Zs_union, 1)
                area_union = np.sum(Zs_union) * cell**2

                # area_union = np.count_nonzero(Zs_union) * cell**2

                Zs_inters = np.minimum(Zs1, Zs2)

                Zs_inters = Zs_inters / np.maximum(Zs_inters, 1)
                area_inters = np.sum(Zs_inters) * cell**2

                # area_inters = np.count_nonzero(Zs_inters) * cell**2
                fitting_parameter = area_inters / area_union

                print('--------------------------------')
                print('With masking threshold', masking_threshold[i_thr])
                print('Union area', area_union, 'Intersect. area', area_inters)
                print('Fitting parameter', fitting_parameter)

                Zs1_mean = np.mean(Zs1*Zs_inters) * nx*ny / \
                    np.count_nonzero(Zs_inters)
                Zs2_mean = np.mean(Zs2*Zs_inters) * nx*ny / \
                    np.count_nonzero(Zs_inters)

                Zs1_vol = Zs1_mean * area_inters
                Zs2_vol = Zs2_mean * area_inters

                print('Vol 1 in intersect.', Zs1_vol, 'Vol 2 in intersect.',
                      Zs2_vol)

                Zs_diff = np.abs(Zs1 - Zs2)

                Zs_diff = Zs_diff * Zs_inters

                avg_thick_diff = np.mean(Zs_diff) * \
                    nx*ny / np.count_nonzero(Zs_inters)
                std_thick_diff = np.std(Zs_diff) * nx * \
                    ny / np.count_nonzero(Zs_inters)
                vol_diff = avg_thick_diff * area_inters

                rel_err_vol = vol_diff / np.maximum(Zs1_vol, Zs2_vol)

                print('Thickness relative error', rel_err_vol)
                print('--------------------------------')

    output_dist = run_name + '_dist_full.asc'

    # ST skipped this (and conjugated masked) to save up disk
    # space (poorly used so far):
    """
    np.savetxt(output_dist,
               np.flipud(Zdist),
               header=header,
               fmt='%4i',
               comments='')

    print('')
    print(output_dist + ' saved')

    output_dist = run_name + '_dist_masked.asc'

    if (masking_threshold < 1):

        Zdist = (1 - masked_Zflow.mask) * Zdist + masked_Zflow.mask * 0

        np.savetxt(output_dist,
                   np.flipud(Zdist),
                   header=header,
                   fmt='%4i',
                   comments='')

        print('')
        print(output_dist + ' saved')

    """

    if (hazard_flag):

        output_haz = run_name + '_hazard_full.asc'

        np.savetxt(output_haz,
                   np.flipud(Zhazard),
                   header=header,
                   fmt='%1.5f',
                   comments='')

        print('')
        print(output_haz + ' saved')

        for i_thr in range(n_masking):

            if (masking_threshold[i_thr] < 1):

                max_Zhazard = int(np.floor(np.max(Zhazard)))

                total_Zflow = np.sum(Zflow)

                # for i in range(1,max_Zhazard):
                for i in np.unique(Zhazard):

                    masked_Zflow = ma.masked_where(Zhazard < i, Zflow)

                    if (flag_threshold == 1):

                        volume_fraction = np.sum(masked_Zflow) / total_Zflow

                        coverage_fraction = volume_fraction

                    else:

                        area_fraction = np.true_divide(
                            np.sum(masked_Zflow > 0), np.sum(Zflow > 0))

                        coverage_fraction = area_fraction

                    if (coverage_fraction < masking_threshold):

                        break

                output_haz_masked = run_name + '_hazard_masked' + '_' + \
                    str(masking_threshold[i_thr]).replace('.', '_') + '.asc'

                np.savetxt(output_haz_masked,
                           np.flipud((1 - masked_Zflow.mask) * Zhazard),
                           header=header,
                           fmt='%1.5f',
                           comments='')

                print('')
                print(output_haz_masked + ' saved')

    # this is to save an additional output for the cumulative deposit,
    # if restart_files is not empty load restart files (if existing)
    if (len(restart_files) > 0):
        for i_restart in range(0, len(restart_files)):

            Zflow_old = np.zeros((ny, nx))

            source = restart_files[i_restart]

            file_exists = exists(source)
            if not file_exists:
                print(source + ' not found.')
                quit()

            hdr = [getline(source, i) for i in range(1, 7)]
            try:
                values = [float(h.split(" ")[-1].strip()) for h in hdr]
            except ValueError:
                print("An problem occurred with header of file ", source)
                print(hdr)

            nd = values[5]

            # Load the previous flow thickness into a numpy array
            arr = np.loadtxt(source, skiprows=6)
            arr[arr == nd] = 0.0

            Zflow_old = np.flipud(arr)

            if crop_flag:

                Zflow_old = Zflow_old[jS:jN, iW:iE]

            Zflow = Zflow + Zflow_old

        output_full = run_name + '_thickness_cumulative.asc'

        np.savetxt(output_full,
                   np.flipud(Zflow),
                   header=header,
                   fmt='%1.5f',
                   comments='')

        output_thickness_cumulative = run_name + '_thickness_cumulative.asc'

        print('')
        print(output_thickness_cumulative + ' saved')
