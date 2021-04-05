
# name of the run (used to save the parameters and the output)
run_name = 'example_run'
# File name of ASCII digital elevation model
source = "./DEM/topography.asc"


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
# vent_flag = 4  => the initial lobes are on multiple
#                   fissures and all the point of the fissures
#                   have the same probability
# vent_flag = 5  => the initial lobes are on multiple
#                   fissures and all the fissures
#                   have the same probability
# vent_flag = 6  => the initial lobes are on the polyline connecting
#                   the vents and the probability of
#                   each segment is fixed by "fissure probabilities"
# vent_flag = 7  => the initial lobes are on multiple
#                   fissures and the probability of
#                   each fissure is fixed by "fissure_probabilities"

vent_flag = 1 

#vent coordinates
x_vent = [ 436358 ]
y_vent = [ 326012 ]

# this syntax define a fissure
#x_vent = [ 332255 , 333694 ]
#y_vent = [ 378125 , 379514 ]

# this coordinates are used when multiple fissures are defined:
# the first one goes from (x_vent[0],y_vent[0]) to (x_vent_end[0],y_vent_end[0])
# the second one goes from (x_vent[1],y_vent[1]) to (x_vent_end[1],y_vent_end[1])
#x_vent_end = [ 332755 , 334194 ]
#y_vent_end = [ 378125 , 379514 ]

# this values defines the probabilities of the different segments of the polyline 
# or of the different fissures.
#fissure_probabilities = [ 2 , 1 ]

# If this flag is set to 1 then a raster map is saved where the values
# represent the probability of a cell to be covered. 
hazard_flag = 1

# Fraction of the volume emplaced or the area invaded (according to the flag
# flag_threshold) used to save the run_name_*_masked.asc files.
# In this way we cut the "tails" with a low thickness (*_thickness_masked.asc)
# and a low probability (*_hazard_masked.asc). The file is used also
# to check the convergence of the solutions increasing the number of flows.
# The full outputs are saved in the files run_name_*_full.asc
# The masked files are saved only when masking_thresold < 1.
masking_threshold = 0.96

# Number of flows
n_flows = 700

# Minimum number of lobes generated for each flow
min_n_lobes = 1200

# Maximum number of lobes generated for each flow
max_n_lobes = min_n_lobes

# If volume flag = 1 then the total volume is read in input, and the
# thickness or the area of the lobes are evaluated according to the
# flag fixed_dimension_flag and the relationship V = n*area*T.
# Otherwise the thickness and the area are read in input and the total
# volume is evaluated (V = n*area*T).
volume_flag = 1

# Total volume (this value is used only when volume_flag = 1) set "1" to be confirmed.
total_volume = 650000000  # m^3

# This flag select which dimension of the lobe is fixed when volume_flag=1:
# fixed_dimension_flag = 1  => the area of the lobes is assigned
# fixed_dimension_flag = 2  => the thickness of the lobes is assigend
fixed_dimension_flag = 2

# Area of each lobe ( only effective when volume_flag = 0 or fixed_dimension_flag = 1 )
lobe_area = 900   # m^2 

# Thickness of each lobe ( only effective when volume_flag = 0 or fixed_dimension_flag  2 )
avg_lobe_thickness = 0.07

# Ratio between the thickness of the first lobe of the flow and the thickness of the
# last lobe.
# thickness_ratio < 1   => the thickness increases with lobe "age"
# thickness_ratio = 1   => all the lobes have the same thickness
# thickness_ratio > 1   => the thickness decreases with lobe "age"
thickness_ratio = 2

# This flag controls if the topography is modified by the lobes and if the
# emplacement of new flows is affected by the changed slope
# topo_mod_flag = 0   => the slope does not changes
# topo_mod_flag = 1   => the slope is re-evaluated every n_flows_counter flows
# topo_mod_flag = 2   => the slope is re-evaluated every n_lobes_counter flows
#                        and every n_flows_counter flows
topo_mod_flag = 1

# This parameter is only effective when topo_mod_flag = 1 and defines the
# number of flows for the re-evaluation of the slope modified by the flow
n_flows_counter = 1

# This parameter is only effective when topo_mod_flag = 2 and defines the
# number of lobes for the re-evaluation of the slope modified by the flow
#n_lobes_counter = 500

# This parameter (between 0 and 1) allows for a thickening of the flow giving
# controlling the modification of the slope due to the presence of the flow.
# thickening_parameter = 0  => minimum thickening (maximum spreading)
# thickening_parameter = 1  => maximum thickening produced in the output
# default thickening_parameter = 0.2
# if you reduce this, the impact of the lava flow is lessened in the computation of the slope, 
# but the thickness is still correct. this allows for "channel" flow, if = 1, 
# then sublava flow would not happen. 
thickening_parameter = 0.7

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
lobe_exponent = 0.1


# max_slope_prob is related to the probability that the direction of 
# the new lobe is close to the maximum slope direction:
# max_slope_prob = 0 => all the directions have the same probability;
# max_slope_prob > 0 => the maximum slope direction has a larger 
#                       probaiblity, and it increases with increasing 
#			value of the parameter;
# max_slope_prob = 1 => the direction of the new lobe is the maximum
#			slope direction.
max_slope_prob = 0.7

# Inertial exponent: 
# inertial_exponent = 0 => the max probability direction for the new lobe is the
#                          max slope direction;
# inertial_exponent > 0 => the max probability direction for the new lobe takes 
#                          into account also the direction of the parent lobe and 
#                          the inertia increaes with increasing exponent
inertial_exponent = 0.1 



