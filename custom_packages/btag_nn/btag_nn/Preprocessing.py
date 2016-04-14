'''
---------------------------------------------------------------------------------------
Preprocessing code for training with Keras

Includes: Data selection
          Splitting of training and testing set
          2D reweighting in eta and pT
          Resetting of default variable value
          Addition of binary check variables for variable categories (SV1, etc.)
---------------------------------------------------------------------------------------
'''

import os
import sys
import array
import numpy as np
import pandas as pd
from pandautils import root2panda
import pickle, json
import argparse


def get_initial_DataFrame(inFile, eta_bins, pt_bins, pid_dict, classes_str):
    """
    This function loads the data.
    In case the input dile 'inFile' is specified, the pandas DataFrame will be constructed from the ROOT input file, cuts will be applied and it will be stored to HDF5 for the next iteration in case the same cuts are to be used but e.g. a different reweighing procedure.
    """
    from btag_nn_inputs import get_jet_eta_str, get_jet_pt_str
    jet_eta_str = get_jet_eta_str()
    jet_pt_str = get_jet_pt_str()

    if inFile:
        print 'Convert ROOT file to pandas DataFrame...'
        df = root2panda('%s' % inFile, 'minibtag')
        print 'conversion complete'
        # only interested in absolute values of eta and the label, so this will speed the calculations up:
        df.update(df[jet_eta_str].abs(), join = 'left', overwrite = True) # only use absolute value of eta
        df.update(df['label'].abs(), join = 'left', overwrite = True) # only use absolute value of labels
        # optional: switch to pT [GeV] ([MeV] by default; keep this consistent with the pT-bins in above code!):
        df.update(df[jet_pt_str].divide(1000.), join = 'left', overwrite = True)
        # dataset selection: pile-up removal and selection in eta, pT acceptance region, limited to b-, c- and light jets:
        if "tau" in pid_dict:
            df = df[(df['label']==pid_dict.get("b")) | (df['label']==pid_dict.get("c")) | (df['label']==pid_dict.get("u")) | (df['label']==pid_dict.get("tau"))] # jet flavor selection
        else:
            df = df[(df['label']==pid_dict.get("b")) | (df['label']==pid_dict.get("c")) | (df['label']==pid_dict.get("u"))] # jet flavor selection
        df = df[(df[jet_pt_str] > pt_bins[0]) & (df[jet_eta_str] < eta_bins[len(eta_bins)-1])] # jet min-pT and max-abs-eta cut
        df = df[((df['JVT'] > 0.59) & (df[jet_eta_str] < 2.4) & (df[jet_pt_str] < 60.)) | (df[jet_pt_str] >= 60.) | (df[jet_eta_str] >= 2.4)] # pile-up removal (use this when working in GeV)
        # store as HDF5 file to speed up the progress for next iteration:
        df.to_hdf('inputFiles/'+classes_str+'jet_ntuple_pTmax'+str(int(pt_bins[len(pt_bins)-1]))+'.h5','df')
        print 'saved input data in HDF5 format for next run.'
        return df
    elif not inFile:
        try:
            if not os.path.isfile('inputFiles/'+classes_str+'jet_ntuple_pTmax'+str(int(pt_bins[len(pt_bins)-1]))+'.h5'):
                print "File does not exist. Try running the path to the ROOT file as additional argument."
                return False
        except IOError as ex:
            print('({})'.format(e))
        return pd.read_hdf('inputFiles/'+classes_str+'jet_ntuple_pTmax'+str(int(pt_bins[len(pt_bins)-1]))+'.h5','df')


def get_defaults():
    """
    This function provides a map for variable default values as found in the ROOT input file.
    """
    # current names adapted to Olaf's b-tagging mini-ntuple
    defaultA = -1
    defaultB = -90
    defaultC = -19
    defaultD = -9
    names_array_check_defaultA = ['jf_nvtx', 'jf_nvtx1t', 'jf_ntrkv', 'sv1_ntkv', 'sv1_n2t', 'sv1_dR']
    names_array_check_defaultB = ['ip2', 'ip3', 'jf_mass', 'jf_efrc', 'jf_sig3', 'sv1_mass', 'sv1_efrc', 'sv1_Lxy', 'sv1_L3d', 'sv1_sig3']
    names_array_check_defaultC = ['ip2_c', 'ip3_c', 'ip2_cu', 'ip3_cu']
    names_array_check_defaultD = ['jf_dR']
    # build python dictionary:
    default_variables_dict = {x: float(defaultA) for x in names_array_check_defaultA}
    for i in names_array_check_defaultB:
        default_variables_dict[i] = float(defaultB)
    for i in names_array_check_defaultC:
        default_variables_dict[i] = float(defaultC)
    for i in names_array_check_defaultD:
        default_variables_dict[i] = float(defaultD)

    # set data types:
    default_variables_dict_dtype = {x: 'float32' for x in default_variables_dict.keys()}
    for i in default_variables_dict_dtype.keys():
        if len(i.split('_'))>1:
            if i.split('_')[1].startswith('n'):
                default_variables_dict_dtype[i] = 'int'
    return default_variables_dict, default_variables_dict_dtype


def calculate_defaults(df, default_variables_dict, default_variables_dict_dtype):
    """
    This function calculates the new default variable. It's either the mean or for one motivated by physics.
    """
    def get_meanValue(df, variable, default_value, dtype):
        """
        This function calculates the mean value for the variable from the non-default distribution. It also checks if there is a default motivated by physics.
        """
        def check_physics(var):
            """
            This function checks if there is a default motivated by physics.
            """
            physics_reason = False
            physics_motivated_value = 0.
            # exceptions are to be extended (will have a look at the new mc15 samples first)
            if len(var.split('_'))>1:
                if var.split('_')[1]=='efrc':
                    physics_reason = True
                    physics_motivated_value = 0.
                elif var is 'sv1_ntkv':
                    physics_reason = True
                    physics_motivated_value = 2.
            return physics_reason, physics_motivated_value
        physics_reason, physics_motivated_value = check_physics(variable)
        if physics_reason==False:
            if dtype is 'int':
                return int(round(df[df[variable] > default_value][variable].sum()*1./df[df[variable] > default_value][variable].count()))
            else:
                return df[df[variable] > default_value][variable].mean()
        else:
            return physics_motivated_value

    # create dict for mean values (to be calculated; dummy values for now)
    default_variables_dict_mean = {x: -10 for x in default_variables_dict.keys()}
    for variable in default_variables_dict.keys():
        default_variables_dict_mean[variable] = get_meanValue(df, variable, default_variables_dict[variable], default_variables_dict_dtype[variable])
    return default_variables_dict_mean


def reset_defaults(df, default_variables_dict, new_default_value_dict):
    """
    This function resets the variable default values for each jet to the DataFrame and also add the binary variables that indicate missing|physics values.
    """
    def get_variableCategories(default_variables_dict):
        """
        This function sets up the categories for which the binary variables need to be added.
        """
        categoryNames = []
        for category in default_variables_dict:
            categoryName = category.split('_')[0] # calculate the name of the category
            if categoryName not in categoryNames:
                categoryNames.append(categoryName)
        return [x for x in categoryNames if x != '']

    from btag_nn_inputs import append_input_variables
    #categories = get_variableCategories(default_variables_dict)
    categories = append_input_variables([])
    checks = np.zeros((len(df['label']),len(categories)), dtype = np.int)

    # select DataFrame that contains only those jets that do not have any devault values for any of the variables
    df_list_complete = df[default_variables_dict.keys()]
    for variable in default_variables_dict.keys(): # df will get updated/shrinked during this loop
        df_list_complete = df_list_complete[df_list_complete[variable] > default_variables_dict[variable]]
    rowindex_drop_list = df_list_complete.index.tolist() # no need to look at those jets
    del df_list_complete
    # drop the complete jets - no need to add a new default there
    df_i = df
    df_i = df_i.drop(rowindex_drop_list, axis=0) # removing completely defined jets (rows) to shrink dataset and speed up
    # add binary series for categories:
    category_check_incomplete = []
    # list to store the boolean checks, np.zeros sets them all to 'False' by default
    for category_itr, category in enumerate(categories):
        category_check_incomplete.append(np.zeros(df.index[len(df.index)-1]+1, dtype=int))
    for column in df_i.columns.tolist():
        df_j = df_i
        if column in default_variables_dict.keys():
            # shrink DataFrame to one that only contains jets with default values for the given variable 'column'
            df_j = df_j[df_j[column] <= default_variables_dict[column]]
            for category_itr, category in enumerate(categories):
                #if column.startswith(category):
                if column is category:
                    category_check_incomplete[category_itr][df_j[column].index.tolist()]=1
                    break
            if len(df_j[column].index) > 0:
                # replace old defaults with new ones; update aligns on indices:
                df.update(df_j[column].replace(to_replace=df_j.get_value(index=df_j[column].index.tolist()[1], col=column), value=new_default_value_dict[column]), join="left", overwrite=True)
        del df_j
    del df_i
    # adding Series to the DataFrame:
    for category_itr, category in enumerate(categories):
        category_check_incomplete[category_itr] = [category_check_incomplete[category_itr][x] for x in xrange(0,len(category_check_incomplete[category_itr])) if x in df.index]
        # add additional columns to indicate if the category was complete:
        df[category+'_check'] = pd.Series(category_check_incomplete[category_itr], index=df.index)
    return df


def calculate_reweighting_general(df_b, df_c, df_u, df_tau, eta_bins, pt_bins):
    """
    This function performs a bin-content count for each individual flavor which is the first step in the Reweighing procedure.
    """
    from btag_nn_inputs import get_jet_eta_str, get_jet_pt_str
    jet_eta_str = get_jet_eta_str()
    jet_pt_str = get_jet_pt_str()

    bin_counts_b = np.zeros((len(eta_bins)-1, len(pt_bins)-1), dtype = np.int)
    bin_counts_c = np.zeros((len(eta_bins)-1, len(pt_bins)-1), dtype = np.int)
    bin_counts_u = np.zeros((len(eta_bins)-1, len(pt_bins)-1), dtype = np.int)
    bin_counts_tau = np.zeros((len(eta_bins)-1, len(pt_bins)-1), dtype = np.int)
    tau_bool = False
    if len(np.asarray(df_tau['label']))!=0:
        tau_bool = True
    for eta_itr, eta_item in enumerate(eta_bins):
        for pt_itr, pt_item in enumerate(pt_bins):
            if eta_itr is not 0 and pt_itr is not 0:
                bin_counts_b[eta_itr-1][pt_itr-1] = df_b[(df_b[jet_eta_str] > eta_bins[eta_itr-1]) & (df_b[jet_eta_str] <= eta_bins[eta_itr]) & (df_b[jet_pt_str] > pt_bins[pt_itr-1]) & (df_b[jet_pt_str] <= pt_bins[pt_itr])][['label']].size
                bin_counts_c[eta_itr-1][pt_itr-1] = df_c[(df_c[jet_eta_str] > eta_bins[eta_itr-1]) & (df_c[jet_eta_str] <= eta_bins[eta_itr]) & (df_c[jet_pt_str] > pt_bins[pt_itr-1]) & (df_c[jet_pt_str] <= pt_bins[pt_itr])][['label']].size
                bin_counts_u[eta_itr-1][pt_itr-1] = df_u[(df_u[jet_eta_str] > eta_bins[eta_itr-1]) & (df_u[jet_eta_str] <= eta_bins[eta_itr]) & (df_u[jet_pt_str] > pt_bins[pt_itr-1]) & (df_u[jet_pt_str] <= pt_bins[pt_itr])][['label']].size
                if tau_bool:
                    bin_counts_tau[eta_itr-1][pt_itr-1] = df_tau[(df_tau[jet_eta_str] > eta_bins[eta_itr-1]) & (df_tau[jet_eta_str] <= eta_bins[eta_itr]) & (df_tau[jet_pt_str] > pt_bins[pt_itr-1]) & (df_tau[jet_pt_str] <= pt_bins[pt_itr])][['label']].size
    return bin_counts_b, bin_counts_c, bin_counts_u, bin_counts_tau


def calculate_reweighting(reference_spectrum, bin_counts_b, bin_counts_c, bin_counts_u, bin_counts_tau, RW_factors_b, RW_factors_c, RW_factors_u, RW_factors_tau, eta_bins, pt_bins):
    """
    This function calculates the training weights with respect to a reference distribution defined by the variable 'reference_spectrum'
    """
    from btag_nn_inputs import get_jet_eta_str, get_jet_pt_str
    jet_eta_str = get_jet_eta_str()
    jet_pt_str = get_jet_pt_str()

    # choose reference:
    # b-jet reference_spectrum
    if reference_spectrum=="b":
        print "  The reweighting reference is the b distribution."
        reference = bin_counts_b
        correction_factor = 1.

    # (b+c+u)-jet reference_spectrum
    elif reference_spectrum=="bcu":
        print "  The sum of the b-, c- and light-jet spectra will be used as reference distribution."
        reference = np.add(bin_counts_b,bin_counts_c)
        reference = np.add(reference,bin_counts_u)
        correction_factor = 1./3.

    # reweight to pT-flat (b-jet) reference_spectrum
    elif reference_spectrum=="flat_pT":
        print "  The reweighted spectra will be flat in pT for each eta-bin."
        # calculate mean in abs(eta) bins over pT_range:
        means_in_eta_b = bin_counts_b.mean(axis=1)
        reference = means_in_eta_b
        correction_factor = 1.

    for i in range(0,len(eta_bins)-1):
        RW_factors_b[i] = reference[i] *1./(1.*bin_counts_b[i]) * correction_factor
        RW_factors_c[i] = reference[i] *1./(1.*bin_counts_c[i]) * correction_factor
        RW_factors_u[i] = reference[i] *1./(1.*bin_counts_u[i]) * correction_factor
        RW_factors_tau[i] = reference[i] *1./(1.*bin_counts_tau[i]) * correction_factor

    print "  Check reweight-factor range...\n   b-jets:\n    min: ", RW_factors_b[np.isfinite(RW_factors_b)].min(), "\n    max: ", RW_factors_b[np.isfinite(RW_factors_b)].max()
    print "   c-jets:\n    min: ", RW_factors_c[np.isfinite(RW_factors_c)].min(), "\n    max: ", RW_factors_c[np.isfinite(RW_factors_c)].max()
    print "   light-jets:\n    min: ", RW_factors_u[np.isfinite(RW_factors_u)].min(), "\n    max: ", RW_factors_u[np.isfinite(RW_factors_u)].max()
    if not bin_counts_tau.max()==bin_counts_tau.min()==0.:
        print "   tau-jets:\n    min: ", RW_factors_tau[np.isfinite(RW_factors_tau)].min(), "\n    max: ", RW_factors_tau[np.isfinite(RW_factors_tau)].max()

    return RW_factors_b, RW_factors_c, RW_factors_u, RW_factors_tau


def add_reweighBranch(df, data_dict, pid_dict):
    """
    This function adds the training weights for each jet to the DataFrame.
    """
    from btag_nn_inputs import get_jet_eta_str, get_jet_pt_str, get_weight_str
    jet_eta_str = get_jet_eta_str()
    jet_pt_str = get_jet_pt_str()

    AddCandU = True
    sLength = len(df.index)
    weights = np.ones(df.index[sLength-1]+1, dtype = np.float32) # list to store the weights from eta-pT reweighting
    # shrink the DataFrame
    df_prime = df.loc[:,[jet_eta_str, jet_pt_str,'label']]
    for eta_itr, eta_item in enumerate(data_dict['eta_bins']):
        if eta_itr is 0:
            continue
        #print "  ", data_dict['eta_bins'][eta_itr-1], '< jet_eta <=', eta_item
        df_eta = df_prime[(df_prime[jet_eta_str]> data_dict['eta_bins'][eta_itr-1]) & (df_prime[jet_eta_str] <= data_dict['eta_bins'][eta_itr])]
        for pt_itr, pt_item in enumerate(data_dict['pt_bins']):
            if pt_itr is 0:
                continue
            #print "  ",  data_dict['pt_bins'][pt_itr-1], '< jet_pt <=', pt_item
            df_etapt = df_eta[(df_eta[jet_pt_str]> data_dict['pt_bins'][pt_itr-1]) & (df_eta[jet_pt_str]<= data_dict['pt_bins'][pt_itr])]
            for i in df_etapt.index:
                if abs(df_etapt['label'].get_value(i))==pid_dict.get("b"):
                    weights[i] = data_dict['RW_b'][eta_itr-1][pt_itr-1]
                    continue
                elif abs(df_etapt['label'].get_value(i))==pid_dict.get("c"):
                    weights[i] = data_dict['RW_c'][eta_itr-1][pt_itr-1]
                    continue
                elif abs(df_etapt['label'].get_value(i))==pid_dict.get("u"):
                    weights[i] = data_dict['RW_u'][eta_itr-1][pt_itr-1]
                    continue
                elif "tau" in pid_dict:
                    if abs(df_etapt['label'].get_value(i))==pid_dict.get("tau"):
                        weights[i] = data_dict['RW_u'][eta_itr-1][pt_itr-1]
                        continue
    weights = [weights[x] for x in xrange(0,len(weights)) if x in df.index]
    if False in np.isfinite(weights):
        print "ERROR: there are not a Number, positive infinity or negative infinity values in the weights."
        exit()
    df[get_weight_str()] = pd.Series(weights, index=df.index)
    return df
