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
import numpy as np
import pandas as pd


def get_initial_DataFrame(inFile, TTree_name_arr, eta_bins, pt_bins, pid_dict, classes_str):
    """
    This function loads the data.
    In case the input file 'inFile' is specified, the pandas DataFrame will be constructed from the ROOT input file, cuts will be applied and it will be stored to HDF5 for the next iteration in case the same cuts are to be used but e.g. a different reweighing procedure.
    """
    import os
    from btag_nn_inputs import jet_eta_str, jet_pt_str, default_sample_info

    if inFile:

        from numpy.lib.recfunctions import stack_arrays
        from root_numpy import root2rec

        print 'Convert ROOT file to pandas DataFrame...'
        for i in range(len(TTree_name_arr)):
            if i==0:
                df = pd.DataFrame(stack_arrays([root2rec(inFile, TTree_name_arr[i])]))
            else:
                df = df.append(pd.DataFrame(stack_arrays([root2rec(inFile, TTree_name_arr[i])])), ignore_index=True)
        print 'conversion complete'
        # only interested in absolute values of eta and the label, so this will speed the calculations up:
        df.update(df[jet_eta_str].abs(), join = 'left', overwrite = True) # only use absolute value of eta
        df.update(df['label'].abs(), join = 'left', overwrite = True) # only use absolute value of labels
        # dataset selection: pile-up removal and selection in eta, pT acceptance region, limited to b-, c- and light jets:
        if "tau" in pid_dict:
            df = df[(df['label']==pid_dict.get("b")) | (df['label']==pid_dict.get("c")) | (df['label']==pid_dict.get("u")) | (df['label']==pid_dict.get("tau"))] # jet flavor selection
        else:
            df = df[(df['label']==pid_dict.get("b")) | (df['label']==pid_dict.get("c")) | (df['label']==pid_dict.get("u"))] # jet flavor selection
        df = df[(df[jet_pt_str] > pt_bins[0]) & (df[jet_eta_str] < eta_bins[len(eta_bins)-1])] # jet min-pT and max-abs-eta cut
        df = df[((df['JVT'] > 0.59) & (df[jet_eta_str] < 2.4) & (df[jet_pt_str] < 60.)) | (df[jet_pt_str] >= 60.) | (df[jet_eta_str] >= 2.4)] # pile-up removal (use this when working in GeV)
        # store as HDF5 file to speed up the progress for next iteration:
        file_info_str = inFile.split('/')[1].replace('.root','')+'_'+classes_str+'jets_pTmax'+str(int(pt_bins[len(pt_bins)-1])/1000)+'GeV'
        df.to_hdf('inputFiles/'+file_info_str+'.h5','df')
        print 'saved input data in HDF5 format for next run.'
        return df, file_info_str
    elif not inFile:
        file_info_str = default_sample_info+'_'+classes_str+'jets_pTmax'+str(int(pt_bins[len(pt_bins)-1]))
        try:
            if not os.path.isfile('inputFiles/'+file_info_str+'.h5'):
                print "File does not exist. Try running the path to the ROOT file as additional argument."
                return False
        except IOError as ex:
            print('({})'.format(e))
        return pd.read_hdf('inputFiles/'+file_info_str+'.h5','df'), file_info_str


def reset_defaults(df, default_variables_dict, new_default_value_dict):
    """
    This function resets the variable default values for each jet to the DataFrame and also add the binary variables that indicate missing|physics values.
    """
    from btag_nn_inputs import append_input_variables, check_variables

    #categories = get_variableCategories(default_variables_dict)
    variables = append_input_variables([])

    # select DataFrame that contains only those jets that do not have any devault values for any of the variables
    df_list_complete = df[default_variables_dict.keys()]
    for variable in default_variables_dict.keys(): # df will get updated/shrinked during this loop
        df_list_complete = df_list_complete[df_list_complete[variable] > default_variables_dict[variable]]
    rowindex_drop_list = df_list_complete.index.tolist() # no need to look at those jets
    del df_list_complete
    # drop the complete jets - no need to add a new default there
    df_i = df
    df_i = df_i.drop(rowindex_drop_list, axis=0) # removing completely defined jets (rows) to shrink dataset and speed up

    # add binary series for variables:
    var_check_incomplete = []
    # list to store the boolean checks, np.zeros sets them all to 'False' by default
    for var_itr, var in enumerate(check_variables):
        var_check_incomplete.append(np.zeros(df.index[len(df.index)-1]+1, dtype=int))

    for column in df_i.columns.tolist():
        df_j = df_i
        if column in default_variables_dict.keys():
            # shrink DataFrame to one that only contains jets with default values for the given variable 'column'
            df_j = df_j[df_j[column] <= default_variables_dict[column]]
            for var_itr, var in enumerate(check_variables):
                #if column.startswith(var):
                if column==var:
                    var_check_incomplete[var_itr][df_j[column].index.tolist()]=1
                    break
            if len(df_j[column].index) > 0:
                # replace old defaults with new ones; update aligns on indices:
                df.update(df_j[column].replace(to_replace=df_j.get_value(index=df_j[column].index.tolist()[1], col=column), value=new_default_value_dict[column]), join="left", overwrite=True)
        del df_j
    del df_i
    # adding check variable Series to the DataFrame:
    for var_itr, var in enumerate(check_variables):
        var_check_incomplete[var_itr] = [var_check_incomplete[var_itr][x] for x in xrange(0,len(var_check_incomplete[var_itr])) if x in df.index]
        # add additional columns to indicate if the var was complete:
        df[var+'_check'] = pd.Series(var_check_incomplete[var_itr], index=df.index)
    return df


def calculate_reweighting_general(df_b, df_c, df_u, df_tau, eta_bins, pt_bins):
    """
    This function performs a bin-content count for each individual flavor which is the first step in the Reweighing procedure.
    """
    from btag_nn_inputs import jet_eta_str, jet_pt_str

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


def calculate_reweighting(training_reweighting_distribution, bin_counts_b, bin_counts_c, bin_counts_u, bin_counts_tau, RW_factors_b, RW_factors_c, RW_factors_u, RW_factors_tau, eta_bins, pt_bins):
    """
    This function calculates the training weights with respect to a distribution defined by 'training_reweighting_distribution'
    """
    from btag_nn_inputs import jet_eta_str, jet_pt_str

    # choose training reweighting (eta,pT) distribution:
    # b-jet training_reweighting_distribution
    if training_reweighting_distribution=="b":
        print "  The training (eta,pT) reweighting is the b distribution."
        training_reweighting_bin_counts = bin_counts_b
        correction_factor = 1.

    # (b+c+u)-jet training_reweighting_distribution
    elif training_reweighting_distribution=="bcu":
        print "  The sum of the b-, c- and light-jet distributions will be used as training (eta, pT) reweighting distribution."
        training_reweighting_bin_counts = np.add(bin_counts_b,bin_counts_c)
        training_reweighting_bin_counts = np.add(training_reweighting_bin_counts,bin_counts_u)
        correction_factor = 1./3.

    # reweight to pT-flat (b-jet) training_reweighting_distribution
    elif training_reweighting_distribution=="flat_pT":
        print "  The reweighted distributions will be flat in pT for each eta-bin."
        # calculate mean in abs(eta) bins over pT_range:
        means_in_eta_b = bin_counts_b.mean(axis=1)
        training_reweighting_bin_counts = means_in_eta_b
        correction_factor = 1.

    for i in range(0,len(eta_bins)-1):
        RW_factors_b[i] = training_reweighting_bin_counts[i] * 1./(1. * bin_counts_b[i]) * correction_factor
        RW_factors_c[i] = training_reweighting_bin_counts[i] * 1./(1. * bin_counts_c[i]) * correction_factor
        RW_factors_u[i] = training_reweighting_bin_counts[i] * 1./(1. * bin_counts_u[i]) * correction_factor
        RW_factors_tau[i] = training_reweighting_bin_counts[i] * 1./(1. * bin_counts_tau[i]) * correction_factor

    print  "  Check reweight-factor range..."
    valid_weights = True
    if False in np.isfinite(RW_factors_b) or False in np.isfinite(RW_factors_c) or False in np.isfinite(RW_factors_u):
        print "  Not all weights are finite  --> TODO: check binning and/or pT range"
        valid_weights = False

    print "   b-jets:\n    min: ", RW_factors_b[np.isfinite(RW_factors_b)].min(), "\n    max: ", RW_factors_b[np.isfinite(RW_factors_b)].max()
    print "   c-jets:\n    min: ", RW_factors_c[np.isfinite(RW_factors_c)].min(), "\n    max: ", RW_factors_c[np.isfinite(RW_factors_c)].max()
    print "   light-jets:\n    min: ", RW_factors_u[np.isfinite(RW_factors_u)].min(), "\n    max: ", RW_factors_u[np.isfinite(RW_factors_u)].max()

    if not bin_counts_tau.max()==bin_counts_tau.min()==0.:
        print "   tau-jets:\n    min: ", RW_factors_tau[np.isfinite(RW_factors_tau)].min(), "\n    max: ", RW_factors_tau[np.isfinite(RW_factors_tau)].max()
    if RW_factors_b[np.isfinite(RW_factors_b)].max()>=50. or RW_factors_c[np.isfinite(RW_factors_c)].max()>=50. or RW_factors_u[np.isfinite(RW_factors_u)].max()>=50.:
        print "  The weights contain factors > 50. --> TODO: check binning or pT range"
        valid_weights =False

    if not valid_weights:
        for i in range(0,len(eta_bins)-1):
            if RW_factors_b[i].max()>=50. or RW_factors_c[i].max()>=50. or RW_factors_u[i].max()>=50.:
                for j in range(0,len(pt_bins)-1):
                    if RW_factors_b[i][j].max()>=50. or RW_factors_c[i][j].max()>=50. or RW_factors_u[i][j].max()>=50.:
                        print "abs(eta) = ",eta_bins[i]," pt = ", pt_bins[j]
                        print "    RW_factor_b = ", RW_factors_b[i][j]
                        print "    RW_factor_c = ", RW_factors_c[i][j]
                        print "    RW_factor_u = ", RW_factors_u[i][j]
    return RW_factors_b, RW_factors_c, RW_factors_u, RW_factors_tau


def add_reweighBranch(df, info_dict, weight_dict, pid_dict):
    """
    This function adds the training weights for each jet to the DataFrame.
    """
    from btag_nn_inputs import jet_eta_str, jet_pt_str, weight_str


    AddCandU = True
    sLength = len(df.index)
    weights = np.ones(df.index[sLength-1]+1, dtype = np.float32) # list to store the weights from eta-pT reweighting
    # shrink the DataFrame
    df_prime = df.loc[:,[jet_eta_str, jet_pt_str,'label']]
    for eta_itr, eta_item in enumerate(info_dict['jet_eta_bins']):
        if eta_itr is 0:
            continue
        #print "  ", info_dict['jet_eta_bins'][eta_itr-1], '< jet_eta <=', eta_item
        df_eta = df_prime[(df_prime[jet_eta_str]> info_dict['jet_eta_bins'][eta_itr-1]) & (df_prime[jet_eta_str] <= info_dict['jet_eta_bins'][eta_itr])]
        for pt_itr, pt_item in enumerate(info_dict['jet_pt_bins']):
            if pt_itr is 0:
                continue
            #print "  ",  info_dict['jet_pt_bins'][pt_itr-1], '< jet_pt <=', pt_item
            df_etapt = df_eta[(df_eta[jet_pt_str]> info_dict['jet_pt_bins'][pt_itr-1]) & (df_eta[jet_pt_str]<= info_dict['jet_pt_bins'][pt_itr])]
            for i in df_etapt.index:
                if abs(df_etapt['label'].get_value(i))==pid_dict.get("b"):
                    weights[i] = weight_dict.get('b_jet_etapt_weights')[eta_itr-1][pt_itr-1]
                    continue
                elif abs(df_etapt['label'].get_value(i))==pid_dict.get("c"):
                    weights[i] = weight_dict.get('c_jet_etapt_weights')[eta_itr-1][pt_itr-1]
                    continue
                elif abs(df_etapt['label'].get_value(i))==pid_dict.get("u"):
                    weights[i] = weight_dict.get('u_jet_etapt_weights')[eta_itr-1][pt_itr-1]
                    continue
                elif "tau" in pid_dict:
                    if abs(df_etapt['label'].get_value(i))==pid_dict.get("tau"):
                        weights[i] = weight_dict.get('tau_jet_etapt_weights')[eta_itr-1][pt_itr-1]
                        continue
    weights = [weights[x] for x in xrange(0,len(weights)) if x in df.index]
    if False in np.isfinite(weights):
        print "ERROR: there are NaN(s), positive infinity or negative infinity values in the weights."
        exit()
    df[weight_str] = pd.Series(weights, index=df.index)
    return df
