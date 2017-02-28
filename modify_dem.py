import numpy as np
from linecache import getline

source_folder = './DEM/'
source = 'landsea_bef73_a10_east.asc'

# the barrier is defined by a polyline with vertexes x_barrier,y_barrier
x_barrier = [ 436756,437334]
y_barrier = [ 326238,326590]

# these values define the width of the barrier at the vertexes
dist = [ 50 ,50]

# this parameter define the shape of the barrier:
# n = 1 (triangle shape)
# n = 2 (rounded shape)
# n large (the walls become more vertical)
n = 1

# ratio between height and width of the barrier
hw_ratio = 0.2



def p2LSbis(Q1,Q2,P):

    x1 = Q1[0]
    y1 = Q1[1]
    x2 = Q2[0]
    y2 = Q2[1]
    x0 = P[0,:]
    y0 = P[1,:]
    
    check12 = (x2-x1)*(x0-x1)+(y2-y1)*(y0-y1) >= 0
    check2 = (x2-x1)*(x0-x2)+(y2-y1)*(y0-y2) <= 0

        

        
    # Distance of the grid point P from the segment joining Q1 and Q2
    d12 = np.abs((x2-x1)*(y0-y1)-(y2-y1)*(x0-x1))/np.sqrt((x2-x1)**2+(y2-y1)**2)
    
    d2 = np.sqrt((x0-x2)**2+(y0-y2)**2)  # distance P-Q2
    d1 = np.sqrt((x0-x1)**2+(y0-y1)**2)  # distance P_Q1
    
    d = check12 * ( check2 * d12 + ( 1 - check2 ) * d2 ) + ( 1 - check12 ) * d1  
    
    alfa1 = np.sqrt( np.maximum( np.zeros( d1.shape[0] ) , d1**2 - d12**2 ))
    alfa2 = np.sqrt( np.maximum( np.zeros( d2.shape[0] ) , d2**2 - d12**2 ))
    
    alfa = alfa1 / ( alfa1 + alfa2 )

    return (d,alfa) 
    

# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(source_folder+source, i) for i in range(1,7)]
values = [float(h.split(" ")[-1].strip()) \
 for h in hdr]
cols,rows,lx,ly,cell,nd = values

# Load the dem into a numpy array
arr = np.loadtxt(source_folder+source, skiprows=6)

nx = arr.shape[1]
xs = lx + cell*np.arange(0,nx)
xmin = np.min(xs)
xmax = np.max(xs)

ny = arr.shape[0]
ys = ly + cell*np.arange(0,ny)
ymin = np.min(ys)
ymax = np.max(ys)

Zs = np.zeros((ny,nx))

Xs,Ys = np.meshgrid(xs,ys)

for i in range(0,ny):

   Zs[i,0:nx] = arr[i,0:nx]

   
X_1d = Xs.ravel()
Y_1d = Ys.ravel()


P = np.stack((X_1d,Y_1d))

barrier_2d = np.zeros((ny,nx))

for i in range(0,len(x_barrier)-1):
    
    Q1 = [ x_barrier[i], y_barrier[i]]
    Q2 = [ x_barrier[i+1], y_barrier[i+1]]


    [dist_1d,alfa_1d] = p2LSbis(Q1,Q2,P)

    dist_2d = dist_1d.reshape(ny,nx)

    alfa_dist = alfa_1d * dist[i] + ( 1-alfa_1d) * dist[i+1]

    fiss_1d = ( alfa_dist**n - np.minimum( alfa_dist**n , dist_1d**n ) )**(1.0/n)

    fiss_2d = fiss_1d.reshape(ny,nx)

    barrier_2d = np.maximum(barrier_2d,fiss_2d)
    
    
Zmod = Zs + hw_ratio * np.flipud(barrier_2d)

header = "ncols     %s\n" % Zs.shape[1]
header += "nrows    %s\n" % Zs.shape[0]
header += "xllcorner " + str(lx-cell) +"\n"
header += "yllcorner " + str(ly+cell) +"\n"
header += "cellsize " + str(cell) +"\n"
header += "NODATA_value -9999\n"


mod_file = source.strip('.asc')+'_mod.asc'

np.savetxt(mod_file, Zmod, header=header, fmt='%1.5f',comments='')


