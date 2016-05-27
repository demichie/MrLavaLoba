# name of the run (used to save the parameters and the output)
run_name = 'test_Vest' # is the easternmost fissure

# File name of ASCII digital elevation model
source = "./DEM/vest20_bef73.asc"

# this is a list of names of exiting *_thickness_*.asc files to be used if
# you want to start a simulation considering the existence of previous
# flows. Leave the field empy [] if you don't want to use it.
# If you want to use more than one file separate them with a comma.
# Example: restart_file = ['flow_001_thickness_full.asc','flow_002_thickness_full.asc]
restart_files = []

# If saveshape_flag = 1 then the output with the lobes is saved on a shapefile
saveshape_flag = 0

# If saveshape_flag = 1 then the raster output is saved as a *.asc file
saveraster_flag = 1

# If this flag is set to 1 then a raster map is saved where the values
# represent the probability of a cell to be covered. 
hazard_flag = 1

# Flag to select if it is cutted the volume of the area
# flag_threshold = 1  => volume
# flag_threshold = 2  => area
flag_threshold = 1

# Fraction of the volume emplaced or the area invaded (according to the flag
# flag_threshold) used to save the run_name_*_masked.asc files.
# In this way we cut the "tails" with a low thickness (*_thickness_masked.asc)
# and a low probability (*_hazard_masked.asc). The file is used also
# to check the convergence of the solutions increasing the number of flows.
# The full outputs are saved in the files run_name_*_full.asc
# The masked files are saved only when masking_thresold < 1.
masking_threshold = 0.97

# If plot_lobes_flag = 1 then all the lobes generated are plotted
plot_lobes_flag = 0

# If plot_lobes_flag = 1 then all the lobes generated are plotted
plot_flow_flag = 0

# Number of flows
n_flows = 4#600

# Minimum number of lobes generated for each flow
# if you increase this, and keep everything else constant, the lava advances farther
min_n_lobes = 100#1000 # changed from 1500 5-1-2016 MAP to give some uncertainty

# Maximum number of lobes generated for each flow
max_n_lobes = 1500

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
max_length = 10

# If volume flag = 1 then the total volume is read in input, and the
# thickness or the area of the lobes are evaluated according to the
# flag fixed_dimension_flag and the relationship V = n*area*T.
# Otherwise the thickness and the area are read in input and the total
# volume is evaluated (V = n*area*T).

# You either choose to input Volume OR Area and Thickness.

volume_flag = 1

# Total volume (this value is used only when volume_flag = 1) set "1" to be confirmed.
total_volume = 1000000 #1000000000  # m^3

# This flag select which dimension of the lobe is fixed when volume_flag=1:
# fixed_dimension_flag = 1  => the area of the lobes is assigned
# fixed_dimension_flag = 2  => the thickness of the lobes is assigend
fixed_dimension_flag = 1

# Area of each lobe ( only effective when volume_flag = 0 or fixed_dimension_flag = 1 )
lobe_area = 4000   # m^2 MAP- right now this is 20m X 20m which is the resolution of the DEM

# Thickness of each lobe ( only effective when volume_flag = 0 or fixed_dimension_flag  2 )
avg_lobe_thickness = 0.002   # m

# Ratio between the thickness of the first lobe of the flow and the thickness of the
# last lobe.
# thickness_ratio < 1   => the thickness increases with lobe "age"
# thickness_ratio = 1   => all the lobes have the same thickness
# thickness_ratio > 1   => the thickness decreases with lobe "age"
thickness_ratio = 1



# Number of repetitions of the first lobe (useful for initial spreading)
# this is because you want to control where the spreading is.  
# a tree with many branches, or a tree with few long branches
n_init = 1

# This flag controls if the topography is modified by the lobes and if the
# emplacement of new flows is affected by the changed slope
# topo_mod_flag = 0   => the slope does not changes
# topo_mod_flag = 1   => the slope is re-evaluated every n_flows_counter flows
# topo_mod_flag = 2   => the slope is re-evaluated every n_lobes_counter flows
#                        and every n_flows_counter flows
topo_mod_flag = 1

# This parameter is only effective when topo_mod_flag = 1 and defines the
# number of flows for the re-evaluation of the slope modified by the flow
n_flows_counter = 10

# This parameter is only effective when topo_mod_flag = 2 and defines the
# number of lobes for the re-evaluation of the slope modified by the flow
n_lobes_counter = 1000000000


# This parameter is to avoid that a chain of loop get stuck in a hole. It is
# active only when n_check_loop>0. When it is greater than zero the code
# check if the last n_check_loop lobes are all in a small box. If this is the
# case then the slope modified by flow is evaluated, and the hole is filled.
n_check_loop = 0

# This parameter (between 0 and 1) allows for a thickening of the flow giving
# controlling the modification of the slope due to the presence of the flow.
# thickening_parameter = 0  => minimum thickening (maximum spreading)
# thickening_parameter = 1  => maximum thickening produced in the output
# this is how viscous your lava is. 
thickening_parameter = 0.9

# This flag controls which lobes have larger probability:
# start_from_dist_flag = 1  => the lobes with a larger distance from
#			       the vent have a higher probability
# start_form_dist_flag = 0  => the younger lobes have a higher
# 			       probability
start_from_dist_flag = 0

# Lobe_exponent is associated to the probability that a new lobe will
# be generated by a young or old (far or close to the vent when the
# flag start_from_dist_flag=1) lobe. The closer is lobe_exponent to 0 the 
# larger is the probability that the new lobe will be generated from a 
# younger lobe.
# lobe_exponent = 1  => there is a uniform probability distribution
# 			assigned to all the existing lobes for the choice 
#			of the next lobe from which a new lobe will be 
#			generated. 
# lobe_exponent = 0  => the new lobe is generated from the last one.
lobe_exponent = 0

# max_slope_prob is related to the porbability that the direction of 
# the new lobe is close to the maximum slope direction:
# max_slope_prob = 0 => all the directions have the same probability;
# max_slope_prob > 0 => the maximum slope direction has a larger 
#                       probaiblity, and it increases with increasing 
#			value of the parameter;
# max_slope_prob = 1 => the direction of the new lobe is the maximum
#			slope direction.
max_slope_prob = 0.96


# Inertial exponent: 
# inertial_exponent = 0 => the max probability direction for the new lobe is the
#                          max slope direction;
# inertial_exponent > 0 => the max probability direction for the new lobe takes 
#                          into account also the direction of the parent lobe and 
#                          the inertia increaes with increasing exponent
inertial_exponent = 0.0

# This factor is to choose where the center of the new lobe will be:
# dist_fact = 0  => the center of the new lobe is on the border of the
# 	 	    previous one;
# dist fact > 0  => increase the distance of the center of the new lobe
# 		    from the border of the previous one;
# dist_fact = 1  => the two lobes touch in one point only.
dist_fact = 1.0

# This parameter affect the shape of the lobes. The larger is this parameter
# the larger is the effect of a small slope on the eccentricity of the lobes:
# aspect_ratio_coeff = 0  => the lobe is always a circle
# aspect_ratio_coeff > 0  => the lobe is an ellipse, with the aspect ratio
#			     increasing with the slope 
aspect_ratio_coeff = 20.0

# Maximum aspect ration of the lobes 
max_aspect_ratio = 5.0

# Number of points for the lobe profile
npoints = 30

# This flag select how multiple initial coordinates are treated:
# vent_flag = 0  => the initial lobes are on the vents coordinates
#                   and the flows start initially from the first vent,
#                   then from the second and so on.
# vent_flag = 1  => the initial lobes are chosen randomly from the vents
#                   coordinates and each vent has the same probability
# vent_flag = 2  => the initial lobes are on the polyline connecting
#                   the vents and all the point of the polyline
#                   have the same probability
# vent_flag = 3  => the initial lobes are on the polyline connecting
#                   the vents and all the segments of the polyline
#                   have the same probability
# this is important for fissures!!!

vent_flag = 1 # i will need 3 for the probability simulations

#x_vent = [ 602523 , 602857 ]
#y_vent = [ 486994 , 488375 ]
#x-isn93	y-isn93

#x_vent = [602923.76 , 602584.90 , 602332.57 ]
#y_vent = [488342.51 , 487203.40 , 486662.68 ]

x_vent = [437560 ]
y_vent = [325898 ]


# Shapefile name (use '' if no shapefile is present)
# shape_name = 'colata_1999_west'
shape_name = ''


