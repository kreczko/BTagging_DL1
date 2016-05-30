#!/usr/bin/env python
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
import numpy as np
import pandas as pd
import json
import argparse
from btag_nn import get_initial_DataFrame, reset_defaults, calculate_reweighting_general, calculate_reweighting, add_reweighBranch
from btag_nn_inputs import calculate_defaults


def _run():
    """
    Main program.
    Will perform the following:
    - retrieve arguments (via argparse)
    - define the eta and pT bins used for reweighting and the range of the training set
    - store the indices for splitting into training and testing to HDF5
    - calculate the reweigh weights from complete data set in the previously defined ranges for eta and pT
    - add reweighting weight values to the training set
    - reset default varaibles and add binary variables for each category
    - store pandas DataFrame to HDF5
    - store other information (e.g. default values, binning) to JSON file
    """
    from btag_nn_inputs import jet_pt_str, jet_eta_str, get_jet_pt_bins, jet_eta_bins, flavor_dict
    args = _get_args()
    os.system('mkdir -p PreparedFiles/')

    # define eta, pT binning and therefore the resulting ranges
    jet_pt_bins = get_jet_pt_bins(args.max_pT)
    jet_eta_bins = jet_eta_bins

    flavor_dict = {
        "b":5,
        "c":4,
        "u":0
    }
    if args.classes=="bcut":
        flavor_dict.update({"tau": 15})

    print "Load pandas data frame..."
    # get data either from ROOT MC sample or already selection from previous run (will be checked for max pT)
    df_all, sample_info_str = get_initial_DataFrame(args.input_file, args.input_file_TTree, jet_eta_bins, jet_pt_bins, flavor_dict, args.classes)

    # string used to construct filenames in the following
    out_str = 'PreparedSample__'+sample_info_str+'_TrainFrac'+str(int(args.training_fraction*100))


    print 'done.\nDetermining training and testing split...'

    # check the content after applying data selection cuts
    N_bJets = df_all[(df_all['label']==flavor_dict.get("b"))]['label'].size
    N_cJets = df_all[(df_all['label']==flavor_dict.get("c"))]['label'].size
    N_uJets = df_all[(df_all['label']==flavor_dict.get("u"))]['label'].size
    N_Jets = df_all['label'].size
    NJets_dict = {
        "N_b": N_bJets,
        "N_c": N_cJets,
        "N_u": N_uJets,
        "N": N_Jets,
    }
    if args.classes=="bcut":
        N_tauJets = df_all[(df_all['label']==flavor_dict.get("tau"))]['label'].size
        NJets_dict.update({"N_tau": N_tauJets})

    if args.classes=="bcut":
        del N_tauJets

    # select only jets for training in the eta-pT range used for reweighing
    N_bJets = df_all[(df_all['label']==flavor_dict.get("b")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])]['label'].size
    N_cJets = df_all[(df_all['label']==flavor_dict.get("c")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])]['label'].size
    N_uJets = df_all[(df_all['label']==flavor_dict.get("u")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])]['label'].size
    if args.classes=="bcut":
        N_tauJets = df_all[(df_all['label']==flavor_dict.get("tau")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])]['label'].size

    # TRAINING | TESTING SEPARATION:
    # define training fraction and split off training set:
    training_fraction = args.training_fraction # default: 0.95

    N_bJets_training = int(round(N_bJets * training_fraction))
    N_cJets_training = int(round(N_cJets * training_fraction))
    N_uJets_training = int(round(N_uJets * training_fraction))
    if args.classes=="bcut":
        N_tauJets_training = int(round(N_tauJets * training_fraction))

    jet_list_b_training = df_all[(df_all['label']==flavor_dict.get("b")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][:N_bJets_training]['label'].index.tolist()
    jet_list_c_training = df_all[(df_all['label']==flavor_dict.get("c")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][:N_cJets_training]['label'].index.tolist()
    jet_list_u_training = df_all[(df_all['label']==flavor_dict.get("u")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][:N_uJets_training]['label'].index.tolist()
    jet_list_training = jet_list_b_training+jet_list_c_training+jet_list_u_training
    if args.classes=="bcut":
        jet_list_tau_training = df_all[(df_all['label']==flavor_dict.get("tau")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][:N_tauJets_training]['label'].index.tolist()
        jet_list_training += jet_list_tau_training
        del jet_list_tau_training
    del jet_list_b_training, jet_list_c_training, jet_list_u_training

    jet_list_b_testing = df_all[(df_all['label']==flavor_dict.get("b")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][N_bJets_training:]['label'].index.tolist()
    jet_list_c_testing = df_all[(df_all['label']==flavor_dict.get("c")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][N_cJets_training:]['label'].index.tolist()
    jet_list_u_testing = df_all[(df_all['label']==flavor_dict.get("u")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][N_uJets_training:]['label'].index.tolist()
    # use also jets with higher pT in testing:
    jet_list_b_testing_highpt = df_all[(df_all['label']==5) & (df_all[jet_pt_str]>jet_pt_bins[len(jet_pt_bins)-1])]['label'].index.tolist()
    jet_list_c_testing_highpt = df_all[(df_all['label']==flavor_dict.get("c")) & (df_all[jet_pt_str]>jet_pt_bins[len(jet_pt_bins)-1])]['label'].index.tolist()
    jet_list_u_testing_highpt = df_all[(df_all['label']==0) & (df_all[jet_pt_str]>jet_pt_bins[len(jet_pt_bins)-1])]['label'].index.tolist()
    jet_list_b_testing.extend(jet_list_b_testing_highpt)
    jet_list_c_testing.extend(jet_list_c_testing_highpt)
    jet_list_u_testing.extend(jet_list_u_testing_highpt)
    jet_list_testing = jet_list_b_testing+jet_list_c_testing+jet_list_u_testing
    if args.classes=="bcut":
        jet_list_tau_testing = df_all[(df_all['label']==flavor_dict.get("tau")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])][N_tauJets_training:]['label'].index.tolist()
        jet_list_tau_testing_highpt = df_all[(df_all['label']==flavor_dict.get("tau")) & (df_all[jet_pt_str]>jet_pt_bins[len(jet_pt_bins)-1])]['label'].index.tolist()
        jet_list_tau_testing.extend(jet_list_tau_testing_highpt)
        jet_list_testing += jet_list_tau_testing
        del jet_list_tau_testing
    del jet_list_b_testing, jet_list_c_testing, jet_list_u_testing



    print 'done.\nReweighting...'
    RW_factors_b = np.ones((len(jet_eta_bins)-1, len(jet_pt_bins)-1), dtype = np.float32)
    RW_factors_c = np.ones((len(jet_eta_bins)-1, len(jet_pt_bins)-1), dtype = np.float32)
    RW_factors_u = np.ones((len(jet_eta_bins)-1, len(jet_pt_bins)-1), dtype = np.float32)
    RW_factors_tau = np.ones((len(jet_eta_bins)-1, len(jet_pt_bins)-1), dtype = np.float32)
    if args.reweighting:
        bin_counts_b, bin_counts_c, bin_counts_u, bin_counts_tau = calculate_reweighting_general(df_all[(df_all['label']==flavor_dict.get("b")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])],
                                                                                                  df_all[(df_all['label']==flavor_dict.get("c")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])],
                                                                                                  df_all[(df_all['label']==flavor_dict.get("u")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])],
                                                                                                  df_all[(df_all['label']==flavor_dict.get("tau")) & (df_all[jet_pt_str]<=jet_pt_bins[len(jet_pt_bins)-1])],
                                                                                                  jet_eta_bins, jet_pt_bins)
        print "  Loaded reweighting data.\n  Calculating event weights..."
        RW_factors_b, RW_factors_c, RW_factors_u, RW_factors_tau = calculate_reweighting(args.reweighting,
                                                                                         bin_counts_b, bin_counts_c, bin_counts_u, bin_counts_tau,
                                                                                         RW_factors_b, RW_factors_c, RW_factors_u, RW_factors_tau,
                                                                                         jet_eta_bins, jet_pt_bins)
    elif not args.reweighting:
        print "  ... you chose to not calculate weights. Will be set to one for each jet."

    # use temporary dictionaries for structure:
    info_dict = {
        'jet_eta_bins': list(jet_eta_bins),
        'jet_pt_bins': list(jet_pt_bins),
        'reweighting': args.reweighting,
    }
    info_dict.update({"jet statistics": NJets_dict})
    weight_dict = {
        'b_jet_etapt_weights': RW_factors_b,
        'c_jet_etapt_weights': RW_factors_c,
        'u_jet_etapt_weights': RW_factors_u,
    }
    if 'bcut' in args.classes=="bcut":
        weight_dict.update({'taujet_etapt_weights': RW_factors_tau})

    print '  done.\n  Adding event weights...'
    df_all = add_reweighBranch(df_all, info_dict, weight_dict, flavor_dict)

    print 'done.\nReset default values:\n  Getting variables and calculate new default values...'
    # set default values:
    initial_default_dict, default_dict = calculate_defaults(df_all)
    print '  done.\n  Resetting default values...'
    df_all = reset_defaults(df_all, initial_default_dict, default_dict)

    print 'done.\nStoring output data in HDF5 file format...'
    # store the final DataFrame as HDF5 file:
    out_str = out_str + "__"+args.reweighting+"_reweighting"
    with pd.HDFStore('PreparedFiles/'+out_str+'.h5') as store:
        store.put("PreparedJet_dataframe", df_all)
        store.put("jet_list_training", pd.DataFrame(data=np.asarray(jet_list_training), columns=['jet_list_training']))
        store.put("jet_list_testing", pd.DataFrame(data=np.asarray(jet_list_testing), columns=['jet_list_testing']))
        store.put("defaults", pd.DataFrame(default_dict.items(), columns=['variable_name', 'default_value']))
        for weights_name, weights in weight_dict.items():
            store.put(weights_name, pd.DataFrame(weights))

    # dump info dictionary to JSON (only for documentation purpose):
    print "done.\nStoring info dictionary to JSON..."
    with open('PreparedFiles/'+out_str+'__info.json', 'wb') as info_output_file:
        json.dump(info_dict, info_output_file, indent=2)

    key_str = "'PreparedJet_dataframe', "
    for key_itr, key in enumerate(weight_dict.keys()):
        if key_itr!=len(weight_dict.keys())-1:
            key_str += "'"+key+"', "
        else:
            key_str += "and '"+key+"'"

    print "all done.\nOutputs:\n --> HDF5 output file: ", 'PreparedFiles/'+out_str+'.h5', "  (keys = "+key_str+")"
    print " --> JSON output file: PreparedFiles/"+out_str+"__info.json  (default_dict, (eta, pT) binning and training reweighting distribution info)"


def _get_args():
    """
    This function provides the arguments for the main function either by using default values or those specified by the user.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    help_input_file = "Input ROOT file that will be converted into (default: %(default)s). Empty string will reload earlier produced file."
    help_max_pT = "Maximum jet-pT in training set (default: %(default)s)."
    help_reweighting = "Reweighting procedure for the training set (default: %(default)s). Options are: i) reweigting to a spectrum: 'b', 'bcu', ii) reweighting to the b-spectrum made flat in pT: 'flat_pT'"
    help_training_fraction = "Fraction of samples used for training (default: %(default)s). "
    help_classes = "Classes used for classification (default: %(default)s). "
    parser.add_argument('input_file', type=str,
                        default="inputFiles/V47full_Akt4EMTo.root",
                        help=help_input_file)
    parser.add_argument('-ttree', '--input_file_TTree', type=str, nargs='+',
                        default=["minibtag;65", "minibtag;64"],
                        help="input file TTree names (default: %(default)s)")
    parser.add_argument('-pt', '--max_pT', type=float, default=300.,
                        choices=[300., 500., 1000., 1500.],
                        help=help_max_pT)
    parser.add_argument('-c', '--classes', type=str, default="bcu",
                        choices=["bcu","bcut"],
                        help=help_classes)
    parser.add_argument('-train', '--training_fraction', type=float, default=0.85,
                        help=help_training_fraction)
    parser.add_argument('-rw', '--reweighting', default="b", choices=["b", "bcu", "flat_pT"],
                        help=help_reweighting)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    _run()
