'''
---------------------------------------------------------------------------------------
Inputs for training with Keras

Includes: Input variable names
          Jet weight string
          Adding check variables
          Calculating new defaults
---------------------------------------------------------------------------------------
'''
default_sample_info = 'V47full_Akt4EMTo'

input_variables = ["ip2", "ip2_c", "ip2_cu", "ip3", "ip3_c", "ip3_cu",
                   "jf_mass", "jf_efrc", "jf_sig3", "jf_nvtx", "jf_nvtx1t", "jf_ntrkv", "jf_n2tv", "jf_dR",
                   "sv1_ntkv", "sv1_mass", "sv1_n2t", "sv1_efrc", "sv1_dR",
                   "sv1_Lxy", "sv1_L3d", "sv1_sig3"]
jet_eta_str = "eta_abs_uCalib"
jet_pt_str = "pt_uCalib"
weight_str = "weights_eta_pT"

check_variables = ['ip2', 'ip3', 'jf_mass', 'sv1_mass']

flavor_dict = {
    "b": 5,
    "c": 4,
    "u": 0,
}

label_dict_Keras = {
    "b": 2,
    "c": 1,
    "u": 0
}

# flat_pT sample from Francesco:
#jet_eta_bins = [0., 0.5, 1., 1.5, 2., 2.5]

# V47:
jet_eta_bins = [0., 0.7, 1.4, 2., 2.25, 2.5]

default_dict = {
    "group A": {
        "variable_name": ['jf_n2tv', 'jf_n2tv','jf_nvtx', 'jf_nvtx1t', 'jf_ntrkv', 'sv1_ntkv', 'sv1_n2t', 'sv1_dR'],
        "initial_value": -1
    },
    "group B": {
        "variable_name": ['ip2', 'ip3', 'jf_mass', 'jf_efrc', 'jf_sig3', 'sv1_mass', 'sv1_efrc', 'sv1_Lxy', 'sv1_L3d', 'sv1_sig3'],
        "initial_value": -90
    },
    "group C": {
        "variable_name": ['ip2_c', 'ip3_c', 'ip2_cu', 'ip3_cu'],
        "initial_value": -19
    },
    "group D": {
        "variable_name": ['jf_dR'],
        "initial_value": -9
    }
}


def append_input_variables(variable_list):
    for var in input_variables:
        variable_list.append(var)
    return variable_list


def get_jet_pt_bins(max_pT):
    import numpy as np

    if max_pT==300.:
        pt_bins = [20., 40., 60., 80., 100., 120., 140., 160., 180., 200., 250., 300.]
    elif max_pT==500.:
        pt_bins = [20., 40., 60., 80., 100., 120., 140., 160., 180., 200., 250., 300., 350., 400., 500.]
    elif max_pT==1000.:
        pt_bins = [20., 40., 60., 80., 100., 120., 140., 160., 180., 200., 250., 300., 350., 400., 500., 600., 700., 800., 1000.]
    elif max_pT==1500.:
        pt_bins = [20., 40., 60., 80., 100., 120., 140., 160., 180., 200., 250., 300., 350., 400., 500., 600., 700., 800., 1000., 1500.]
    pt_bins = np.multiply(pt_bins,1000.)
    return pt_bins


def append_kinematic_variables(variable_list):
    kin_var_arr = []
    kin_var_arr.append(jet_eta_str)
    kin_var_arr.append(jet_pt_str)
    for var in kin_var_arr:
        variable_list.append(var)
    return variable_list


def append_check_variables(variable_list):
    new_variable_list = []
    for var in variable_list:
        new_variable_list.append(var)
        if var in check_variables:
            new_variable_list.append(var+"_check")
    return new_variable_list


def calculate_defaults(df):
    import numpy as np
    """
    Provide a map for variable default values as found in the ROOT input file first.
    """
    # build python dictionary:
    default_variables_dict = {}
    for group_key, group_item in default_dict.iteritems():
        default_variables_dict.update({x: float(group_item.get('initial_value')) for x in group_item.get('variable_name')})

    # set data types:
    default_variables_dict_dtype = {x: 'float32' for x in default_variables_dict.keys()}
    for i in default_variables_dict_dtype.keys():
        if len(i.split('_'))>1:
            if i.split('_')[1].startswith('n'):
                default_variables_dict_dtype[i] = 'int'

    """
    This function calculates the new default variable. It's either the mean or for one motivated by physics.
    """
    def _get_newDefaultValue(df, variable, default_value, dtype):
        """
        This function calculates the new default value for the variable from the non-default distribution. It also checks if there is a default motivated by physics.
        """
        def _check_physics(var):
            """
            This function checks if there is a default motivated by physics.
            """
            physics_reason = False
            physics_motivated_value = 0.
            # exceptions are to be extended (will have a look at the new mc15 samples first)
            if len(var.split('_'))>1:
                if var.split('_')[1]=='efrc' or var=='jf_n2tv':
                    physics_reason = True
                    physics_motivated_value = 0.
                elif var=='sv1_ntkv':
                    physics_reason = True
                    physics_motivated_value = 2.
            return physics_reason, physics_motivated_value

        physics_reason, physics_motivated_value = _check_physics(variable)
        if physics_reason==False:
            if dtype is 'int':
                return int(round(df[df[variable] > default_value][variable].sum()*1./df[df[variable] > default_value][variable].count()))
            else:
                return df[df[variable] > default_value][variable].mean()
        else:
            return physics_motivated_value

    # create dict for new default values (to be calculated)
    default_variables_dict_new = {x: float('nan') for x in default_variables_dict.keys()}
    for variable in default_variables_dict.keys():
        default_variables_dict_new[variable] = _get_newDefaultValue(df, variable, default_variables_dict[variable], default_variables_dict_dtype[variable])
    for variable_value in default_variables_dict_new.values():
        if np.isnan(variable_value):
            import sys, warnings
            warnings.warn("New default values contain NaN.")
            sys.exit(0)
    return default_variables_dict, default_variables_dict_new
