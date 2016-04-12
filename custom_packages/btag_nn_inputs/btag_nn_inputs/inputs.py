'''
---------------------------------------------------------------------------------------
inputs for training with Keras

Includes: Input variable names
          Event weight string
---------------------------------------------------------------------------------------
'''

def append_input_variables(variable_list):
    for var in ["ip2", "ip2_c", "ip2_cu", "ip3", "ip3_c", "ip3_cu",
                "jf_mass", "jf_efrc", "jf_sig3", "jf_nvtx", "jf_nvtx1t", "jf_ntrkv", "jf_n2tv", "jf_dR",
                "sv1_ntkv", "sv1_mass", "sv1_n2t", "sv1_efrc", "sv1_dR",
                "sv1_Lxy", "sv1_L3d", "sv1_sig3"]:
        variable_list.append(var)
    return variable_list


def append_kinematic_variables(variable_list):
    kin_var_arr = []
    kin_var_arr.append(get_jet_eta_str())
    kin_var_arr.append(get_jet_pt_str())
    for var in kin_var_arr:
        variable_list.append(var)
    return variable_list


def append_check_variables(variable_list):
    new_variable_list = []
    for var in variable_list:
        new_variable_list.append(var)
        new_variable_list.append(var+"_check")
    return new_variable_list


def get_jet_eta_str():
    return "eta_abs_uCalib"


def get_jet_pt_str():
    return "pt_uCalib"


def get_weight_str():
    return "weights_eta_pT"
