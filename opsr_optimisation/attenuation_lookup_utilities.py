# Non mitsuba imports
import numpy as np

# %%
# Load the optimised attenuation factors. Note that if you use file that 
# is generated during the optimisation you will also have to specify the optimisation 
# iteration. E.g. : att_factors_2d = np.load('optimisation_script/numpy_data/att_factors_nd_file.npy')[iteration_num]
# /!\ Do not change the naming of those variables as they are used internally by the functions /!\
att_factors_2d = np.load('opsr_interp_data_final/opsr_interp_low_att_factors_2d.npy')
att_factors_3d = np.load('opsr_interp_data_final/opsr_interp_low_att_factors_3d.npy')
att_factors_4d = np.load('opsr_interp_data_final/opsr_interp_low_att_factors_4d.npy')
att_factors_5d = np.load('opsr_interp_data_final/opsr_interp_low_att_factors_5d.npy')

# %%
# The attenuation factors are stored in a 1D array, using reshape, we can modify the coefficients
# corresponding to a specific path more intuitively. For exemple the next line would set all the attenuation
# factors of length 2 of the form LD(X*)E to zero, i.e. all the path ending with a diffuse vertex.
# Note the order from light to the eye ! (This is reversed if compared to the linear_interp_attenuation 
# and attenuation functions defined below)
# The index 0 corresponds to a Specular vertex, the last index (3 in our case) indicate a diffuse vertex.
# In-between, different roughness given our roughness discretisation.  
res = 4 # the number of bins used (the roughness discretisation resolution)
att_factors_2d.reshape([res,res])[-1,:] = 0

# Another exemple setting an SDS path to an attenuation of 0.1
att_factors_3d.reshape([res,res,res])[0,-1,0] = 0.1

# We can also save the result to disk again with np.save
# np.save("some_file_without_extension",att_factors_2d) 

# %%

# Converts from roughness to a bin (remember there are 4 bins)
def roughness_to_bin(vertex_roughness,roughness_res):
    log_roughness_res = roughness_res + 1
    fractional_bins = (2.0**np.sqrt(vertex_roughness) - 1.0) * log_roughness_res
    bin = np.minimum(np.floor(fractional_bins), roughness_res - 1)
    return bin

# Converts from roughness to a fractional bin (for the interpolated version of OPSR) 
def roughness_to_fractional_bin(vertex_roughness,roughness_res):
    log_roughness_res = roughness_res + 1
    return  np.minimum(( 2.0**np.sqrt(vertex_roughness) - 1.0) * log_roughness_res,roughness_res-1)

# Returns the interpolated attenuation factor from a vector of surface roughnesses
# /!\ The input roughnesses go from the eye to the light
def linear_interp_attenuation(roughness_path : np.array):
    subpath_len = len(roughness_path)
    roughness_res = 4

    bins = -np.ones((subpath_len,2))
    fractional_bins = -np.ones(subpath_len)
    for i in range(0,subpath_len):
        bin_float = roughness_to_fractional_bin(roughness_path[i], roughness_res)
        idx_down = np.floor(bin_float)
        idx_up = np.minimum(np.ceil(bin_float), roughness_res-1)
        bins[i] = [idx_down,idx_up]
        fractional_bins[i] = bin_float-idx_down

    # print("fractional_bins : " + str(fractional_bins))

    query_indices = []

    for c in range(0,subpath_len**2):
        query_idx = 0
        for i in range(0,subpath_len):
            # print(bins[i][c>>i & 0b1],c>>i & 0b1)
            query_idx += roughness_res**i * bins[i][c>>i & 0b1]
        query_indices.append(int(query_idx))
    # print("query indices : " + str(query_indices))

    if subpath_len == 2:
        values = att_factors_2d[query_indices]
    elif subpath_len == 3:
        values = att_factors_3d[query_indices]
    elif subpath_len == 4:
        values = att_factors_4d[query_indices]
    elif subpath_len == 5:
        values = att_factors_5d[query_indices]

    # Perform the linear interpolation using the binary trick
    attenuation = 0.0
    for c in range(0,subpath_len**2):
        ratio = 1.0
        for i in range(0,subpath_len):
            ratio *= (1-fractional_bins[i]) if c>>i & 0b1 == 0 else fractional_bins[i]
        attenuation += values[c] * ratio

    attenuation
    return attenuation

# Returns the attenuation factor from a vector of surface roughnesses
# /!\ The input roughnesses go from the eye to the light
def attenuation(roughness_path : np.array):
    subpath_len = len(roughness_path)
    roughness_res = 4

    bins = []
    for i in range(0,subpath_len):
        bins.append(int(roughness_to_bin(roughness_path[i], roughness_res)))

    query_idx = 0
    for i in range(0,subpath_len): 
        query_idx += roughness_res**i * bins[i] 
    # print("query_idx : " + str(query_idx))
    
    if subpath_len == 2:
        return att_factors_2d[query_idx]
    elif subpath_len == 3:
        return att_factors_3d[query_idx]
    elif subpath_len == 4:
        return att_factors_4d[query_idx]
    elif subpath_len == 5:
        return att_factors_5d[query_idx]

# %%
 
# attenuation for EDSL
print(linear_interp_attenuation(np.array([1.0,0.0])))

# attenuation for EDSSL
print(attenuation([1.0,0.0,0.0]))

# To print in a format that can be copied easilly : 
# print(list(att_factors_5d))
# print(list(att_factors_4d))
# print(list(att_factors_3d))
# print(list(att_factors_2d))


# %%
