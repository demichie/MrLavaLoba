def make_plot(X,Y,Z,Z_init,h_min,h_max,simtime,cr_angle,run_name,iter,plot_show_flag):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    import imp

    try:
        imp.find_module('mayavi')
        found = True
    except ImportError:
        found = False
        
    if found:
    
        from mayavi import mlab    

    plt.ion()
    plt.rcParams.update({'font.size':8})

     
    plt.close('all') 
     
    Z_diff = Z-Z_init 
        
    fig = plt.figure()
    fig.set_size_inches(11,7)
    ax1 = fig.add_subplot(2,2,1,projection='3d')
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    plt.tight_layout(pad=4, w_pad=4, h_pad=4)
    time_text = ax3.text(0,0, 'time ='+"{:8.2f}".format(simtime)+'s')
    
    delta_x = X[0,1]-X[0,0]
    delta_y = Y[1,0]-Y[0,0]
    
    Z_x,Z_y = np.gradient(Z,delta_x,delta_y)

    grad_Z = np.sqrt( Z_x**2 + Z_y**2 )

    slope = ( np.arctan(grad_Z)*180.0/np.pi )
    
    my_col = cm.jet(slope/cr_angle)

    norm = colors.Normalize(vmin=0.0, vmax=cr_angle)

    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col , 
                    linewidth=0, antialiased=False, alpha = 0.7)
     
    #m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    #m.set_array([])
    #plt.colorbar(m,ax=ax1,location='bottom') 
                    
    # clb1.set_label('Slope [°]')
                    
                    
    nx = X.shape[1]
    ny = X.shape[0]  
    
    idx1 = int(np.floor(nx/4)) 
    idx2 = int(np.floor(nx/2)) 
    idx3 = int(np.floor(3*nx/4)) 
    
    dh = h_max-h_min
    
    ax1.plot3D(X[idx1,:],Y[idx1,:],Z[idx1,:]+0.01*dh, 'blue')
    ax1.plot3D(X[idx2,:],Y[idx2,:],Z[idx2,:]+0.01*dh, 'red')
    ax1.plot3D(X[idx3,:],Y[idx3,:],Z[idx3,:]+0.01*dh, 'green')

    if ( np.min(Z_diff) == np.max(Z_diff) ):
    
        Z_diff[0,0] = 0.001
        Z_diff[-1,-1] = -0.001
        
   
    ax1.set_xlabel('x [m]')
    ax1.set_xlim(np.amin(X),np.amax(X))
    ax1.set_ylabel('y [m]')
    ax1.set_ylim(np.amin(Y),np.amax(Y))
    ax1.set_zlabel('z [m]')
    ax1.set_zlim(h_min,h_max)

    
    extent = [np.amin(X),np.amax(X),np.amin(Y),np.amax(Y)] 
    
    z_range = np.amax(np.abs(Z_diff))   
    cnt = ax2.imshow(Z_diff, cmap='seismic', extent=extent, vmin=-z_range, vmax=z_range)
    
    clb = plt.colorbar(cnt,ax=ax2)
    clb.set_label('Delta h [m]')
    # colorbar(cnt);
    
    ax2.set_xlabel('x [m]')
    ax2.set_xlim(np.amin(X),np.amax(X))
    ax2.set_ylabel('y [m]')
    ax2.set_ylim(np.amin(Y),np.amax(Y))
    

    
    l1, = ax3.plot(X[idx1,:],Z[idx1,:], 'b-')
    l2, = ax3.plot(X[idx2,:],Z[idx2,:], 'r-')
    l3, = ax3.plot(X[idx3,:],Z[idx3,:], 'g-')
    
    ax3.legend((l1, l2, l3), ("y="+str(Y[idx1,0])+' m',"y="+str(Y[idx2,0])+' m', \
                             "y="+str(Y[idx3,0])+' m'), loc='upper right', shadow=True)

    
    ax3.plot(X[idx1,:],Z_init[idx1,:],'b--')
    ax3.plot(X[idx2,:],Z_init[idx2,:],'r--')
    ax3.plot(X[idx3,:],Z_init[idx3,:],'g--')
           
    ax3.set_xlabel('x [m]')      
    ax3.set_ylabel('z [m]')    
    
    x_min, x_max = ax3.get_xlim()
        
    y_min, y_max = ax3.get_ylim()
    
    time_text.set_position((x_min+0.05*(x_max-x_min), y_min+0.9*(y_max-y_min)))
    time_text.set_text('time ='+"{:8.2f}".format(simtime)+'s')

 
    slope_flat = slope.flatten()
    
    """
    n, bins, patches = ax4.hist(slope_flat, weights=100.0*np.ones(len(slope_flat)) / len(slope_flat),\
             histtype='stepfilled', alpha=0.2)
             
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    cm = plt.cm.get_cmap('RdYlBu_r')
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))   
    """
        
    # Get the histogramp
    cm = plt.cm.get_cmap('jet')

    Yh,Xh = np.histogram(slope_flat,density=True)
    x_span = Xh.max()-Xh.min()
    C = [cm(x/cr_angle) for x in Xh]

    ax4.bar(Xh[:-1]+0.5*(Xh[1]-Xh[0]),Yh,color=C,width=Xh[1]-Xh[0])      

    ax4b = ax4.twinx()
    
    # plot the cumulative histogram
    n_bins = 50
    ax4b.hist(slope_flat, n_bins, density=True, histtype='step',
                           cumulative=True, label='Empirical')

    ax4.set_xlabel('slope [degrees]')
    ax4.set_ylabel('probability density function')
    ax4b.set_ylabel('cumulative distribution function')

    frame_name = run_name + '_{0:03}'.format(iter) + '.png'
    plt.savefig(frame_name,dpi=200)

    frame_name = run_name + '_{0:03}'.format(iter) + '.pdf'
    plt.savefig(frame_name)

    if plot_show_flag:

        plt.show()
        plt.pause(0.01)
    
    return time_text
