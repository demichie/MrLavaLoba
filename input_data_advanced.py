# this is a list of names of exiting *_thickness_*.asc files to be used if
# you want to start a simulation considering the existence of previous
# flows. Leave the field empy [] if you don't want to use it.
# If you want to use more than one file separate them with a comma.
# Example: restart_file = ['flow_001_thickness_full.asc','flow_002_thickness_full.asc]
restart_files = []
# If saveshape_flag = 1 then the output with the lobes is saved on a shapefile
saveshape_flag = 0

# If saveraster_flag = 1 then the raster output is saved as a *.asc file
saveraster_flag = 1

# Flag to select if it is cutted the volume of the area
# flag_threshold = 1  => volume
# flag_threshold = 2  => area
flag_threshold = 1

# If plot_lobes_flag = 1 then all the lobes generated are plotted
plot_lobes_flag = 0

# If plot_lobes_flag = 1 then all the lobes generated are plotted
plot_flow_flag = 0

# The number of lobes of the flow is defined accordingly to a random uniform
# distribution or to a beta law, as a function of the flow number.
# a_beta, b_beta = 0  => n_lobes is sampled randomly in [min_n_lobes,max_n_lobes]
# a_beta, b_beta > 0  => n_lobes = min_n_lobes + 0.5 * ( max_n_lobes - min_n_lobes )
#                                              * beta(flow/n_flows,a_beta,b_beta)
a_beta = 0.0
b_beta = 0.0

# Flag for maximum distances (number of chained lobes) from the vent
force_max_length = 0

# Maximum distances (number of chained lobes) from the vent
# This parameter is used only when force_max_length = 1
max_length = 50


# Number of repetitions of the first lobe (useful for initial spreading)
n_init = 1

# This parameter is to avoid that a chain of loop get stuck in a hole. It is
# active only when n_check_loop>0. When it is greater than zero the code
# check if the last n_check_loop lobes are all in a small box. If this is the
# case then the slope modified by flow is evaluated, and the hole is filled.
n_check_loop = 0

# This flag controls which lobes have larger probability:
# start_from_dist_flag = 1  => the lobes with a larger distance from
#			       the vent have a higher probability
# start_form_dist_flag = 0  => the younger lobes have a higher
# 			       probability
start_from_dist_flag = 0 

# This factor is to choose where the center of the new lobe will be:
# dist_fact = 0  => the center of the new lobe is on the border of the
# 	 	    previous one;
# dist fact > 0  => increase the distance of the center of the new lobe
# 		    from the border of the previous one;
# dist_fact = 1  => the two lobes touch in one point only.
dist_fact = 1

# Number of points for the lobe profile
npoints = 30

# This parameter affect the shape of the lobes. The larger is this parameter
# the larger is the effect of a small slope on the eccentricity of the lobes:
# aspect_ratio_coeff = 0  => the lobe is always a circle
# aspect_ratio_coeff > 0  => the lobe is an ellipse, with the aspect ratio
#			     increasing with the slope 
aspect_ratio_coeff = 2.0

# Maximum aspect ratio of the lobes 
max_aspect_ratio = 2.5

# Shapefile name (use '' if no shapefile is present)
shape_name = ''


