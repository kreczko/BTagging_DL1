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
import pickle, json
import argparse
from btag_nn import get_initial_DataFrame, get_defaults, calculate_defaults, reset_defaults, calculate_reweighting_general, calculate_reweighting, add_reweighBranch


def _run():
    """
    Main program.
    Will perform the following:
    - retrieve arguments (via argparse)
    - define the eta and pT bins used for reweighting and the range of the training set
    - store the indices for splitting into training and testing to HDF5
    - calculate the reweigh weights from complete data set in the previously defined ranges for eta and pT
    - add reweighting branch to the training set
    - reset default varaibles and add binary variables for each category
    - store pandas DataFrame to HDF5
    - store protocol information and other information to pickle file
    """
    from btag_nn_inputs import get_jet_pt_str, get_jet_eta_str
    args = _get_args()
    inFile = args.input_file
    os.system('mkdir -p PreparedFiles/')

    # define eta, pT binning and therefore the resulting ranges
    jet_eta_str = get_jet_eta_str()
    jet_pt_str = get_jet_pt_str()
    if args.max_pT==300.:
        pt_bins = [20.,25., 30.,35., 40.,45., 50.,55., 60.,70.,80., 90.,100., 120.,140., 160.,180., 200.,225.,250.,300.]
    if args.max_pT==500.:
        pt_bins = [20.,25., 30.,35., 40.,45., 50.,55., 60.,70.,80., 90.,100., 120.,140., 160.,180., 200.,225.,250.,300., 350., 400., 500.]
    eta_bins = [0., 0.2, 0.4, 0.6, 0.8, 1.,1.2, 1.4, 1.6, 1.8, 2.,2.2, 2.5]


    label_dict = {
        "b":5,
        "c":4,
        "u":0
    }
    if args.classes=="bcut":
        label_dict.update({"tau": 15})

    # string used to construct filenames in the following
    out_str = 'PreparedSample__'+args.classes+'jet_ntuple_TrainFrac'+str(int(args.training_fraction*100))+'_pTmax'+str(int(pt_bins[len(pt_bins)-1]))

    print "Load pandas data frame..."
    # get data either from ROOT MC sample or already selection from previous run (will be checked for max pT)
    df_all = get_initial_DataFrame(inFile, eta_bins, pt_bins, label_dict, args.classes)
    print 'done.\nDetermining training and testing protocol...'

    # check the content after applying data selection cuts
    N_bJets = df_all[(df_all['label']==label_dict.get("b"))]['label'].size
    N_cJets = df_all[(df_all['label']==label_dict.get("c"))]['label'].size
    N_uJets = df_all[(df_all['label']==label_dict.get("u"))]['label'].size
    N_Jets = df_all['label'].size
    NJets_dict = {
        "N_b": N_bJets,
        "N_c": N_cJets,
        "N_u": N_uJets,
        "N": N_Jets,
    }
    if args.classes=="bcut":
        N_tauJets = df_all[(df_all['label']==label_dict.get("tau"))]['label'].size
        NJets_dict.update({"N_tau": N_tauJets})

    with open("PreparedFiles/"+out_str+"__Njet_info.json",'a') as NJ_info:
        json.dump(NJets_dict, NJ_info, indent=2, sort_keys=True)
    del N_bJets, N_cJets, N_uJets, N_Jets
    if args.classes=="bcut":
        del N_tauJets

    # select only jets for training in the eta-pT range used for reweighing
    N_bJets = df_all[(df_all['label']==label_dict.get("b")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])]['label'].size
    N_cJets = df_all[(df_all['label']==label_dict.get("c")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])]['label'].size
    N_uJets = df_all[(df_all['label']==label_dict.get("u")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])]['label'].size
    if args.classes=="bcut":
        N_tauJets = df_all[(df_all['label']==label_dict.get("tau")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])]['label'].size

    # TRAINING | TESTING SEPARATION:
    # define training fraction and split off training set:
    training_fraction = args.training_fraction # default: 0.80

    N_bJets_training = int(round(N_bJets * training_fraction))
    N_cJets_training = int(round(N_cJets * training_fraction))
    N_uJets_training = int(round(N_uJets * training_fraction))
    if args.classes=="bcut":
        N_tauJets_training = int(round(N_tauJets * training_fraction))

    jet_list_b_training = df_all[(df_all['label']==label_dict.get("b")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][:N_bJets_training]['label'].index.tolist()
    jet_list_c_training = df_all[(df_all['label']==label_dict.get("c")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][:N_cJets_training]['label'].index.tolist()
    jet_list_u_training = df_all[(df_all['label']==label_dict.get("u")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][:N_uJets_training]['label'].index.tolist()
    jet_list_training = jet_list_b_training+jet_list_c_training+jet_list_u_training
    if args.classes=="bcut":
        jet_list_tau_training = df_all[(df_all['label']==label_dict.get("tau")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][:N_tauJets_training]['label'].index.tolist()
        jet_list_training += jet_list_tau_training
        del jet_list_tau_training
    del jet_list_b_training, jet_list_c_training, jet_list_u_training

    jet_list_b_testing = df_all[(df_all['label']==label_dict.get("b")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][N_bJets_training:]['label'].index.tolist()
    jet_list_c_testing = df_all[(df_all['label']==label_dict.get("c")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][N_cJets_training:]['label'].index.tolist()
    jet_list_u_testing = df_all[(df_all['label']==label_dict.get("u")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][N_uJets_training:]['label'].index.tolist()
    # use also jets with higher pT in testing:
    jet_list_b_testing_highpt = df_all[(df_all['label']==5) & (df_all[jet_pt_str]>pt_bins[len(pt_bins)-1])]['label'].index.tolist()
    jet_list_c_testing_highpt = df_all[(df_all['label']==label_dict.get("c")) & (df_all[jet_pt_str]>pt_bins[len(pt_bins)-1])]['label'].index.tolist()
    jet_list_u_testing_highpt = df_all[(df_all['label']==0) & (df_all[jet_pt_str]>pt_bins[len(pt_bins)-1])]['label'].index.tolist()
    jet_list_b_testing.extend(jet_list_b_testing_highpt)
    jet_list_c_testing.extend(jet_list_c_testing_highpt)
    jet_list_u_testing.extend(jet_list_u_testing_highpt)
    jet_list_testing = jet_list_b_testing+jet_list_c_testing+jet_list_u_testing
    if args.classes=="bcut":
        jet_list_tau_testing = df_all[(df_all['label']==label_dict.get("tau")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])][N_tauJets_training:]['label'].index.tolist()
        jet_list_tau_testing_highpt = df_all[(df_all['label']==label_dict.get("tau")) & (df_all[jet_pt_str]>pt_bins[len(pt_bins)-1])]['label'].index.tolist()
        jet_list_tau_testing.extend(jet_list_tau_testing_highpt)
        jet_list_testing = jet_list_tau_testing
        del jet_list_tau_testing
    del jet_list_b_testing, jet_list_c_testing, jet_list_u_testing


    if not os.path.isfile("PreparedFiles/"+out_str+"__jet_lists.h5"):
        print "done.\nStore jet lists to HDF5..."
        df__jet_list_training = pd.DataFrame(data=np.asarray(jet_list_training), columns=['jet_list_training'])
        df__jet_list_testing = pd.DataFrame(data=np.asarray(jet_list_testing), columns=['jet_list_testing'])
        df__jet_list_training.to_hdf('PreparedFiles/'+out_str+'__jet_lists.h5',key='PreparedJet__jet_list_training', mode='w')
        df__jet_list_testing.to_hdf('PreparedFiles/'+out_str+'__jet_lists.h5',key='PreparedJet__jet_list_testing', mode='a')

    print 'done.\nReweighting...'
    RW_factors_b = np.ones((len(eta_bins)-1, len(pt_bins)-1), dtype = np.float32)
    RW_factors_c = np.ones((len(eta_bins)-1, len(pt_bins)-1), dtype = np.float32)
    RW_factors_u = np.ones((len(eta_bins)-1, len(pt_bins)-1), dtype = np.float32)
    RW_factors_tau = np.ones((len(eta_bins)-1, len(pt_bins)-1), dtype = np.float32)
    if args.reweighting:
        bin_counts_b, bin_counts_c, bin_counts_u, bin_counts_tau = calculate_reweighting_general(df_all[(df_all['label']==label_dict.get("b")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])],
                                                                                                  df_all[(df_all['label']==label_dict.get("c")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])],
                                                                                                  df_all[(df_all['label']==label_dict.get("u")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])],
                                                                                                  df_all[(df_all['label']==label_dict.get("tau")) & (df_all[jet_pt_str]<=pt_bins[len(pt_bins)-1])],
                                                                                                  eta_bins, pt_bins)
        print "  Loaded reweighting data.\n  Calculating event weights..."
        RW_b, RW_c, RW_u, RW_tau = calculate_reweighting(args.reweighting,
                                                          bin_counts_b, bin_counts_c, bin_counts_u, bin_counts_tau,
                                                          RW_factors_b, RW_factors_c, RW_factors_u, RW_factors_tau,
                                                          eta_bins, pt_bins)
    elif not args.reweighting:
        RW_b = RW_factors_b
        RW_c = RW_factors_c
        RW_u = RW_factors_u
        RW_tau = RW_factors_tau
        print "  ... you chose not to calculate weights. Will be set to one for each jet."

    print '  done.\n  Adding event weights...'
    data_dict = {
        'eta_bins': eta_bins,
        'pt_bins': pt_bins,
        'reweighting': args.reweighting,
        'RW_b': RW_b,
        'RW_c': RW_c,
        'RW_u': RW_u,
        }
    if 'bcut' in args.input_file:
        data_dict.update({'RW_tau': RW_tau})
    df_all = add_reweighBranch(df_all, data_dict, label_dict)
    print 'done.\nGetting variables and calculating mean values...'

    # set default values:
    init_default_dict, default_variables_dict_dtype = get_defaults()
    default_dict = calculate_defaults(df_all, init_default_dict, default_variables_dict_dtype)
    print 'done.\nResetting default values...'
    df_all = reset_defaults(df_all, init_default_dict, default_dict)
    print 'done.\nStoring output data in HDF5 file format...'
    # store the final DataFrame as HDF5 file:
    out_str = out_str + "__"+args.reweighting+"_reweighting"
    df_all.to_hdf('PreparedFiles/'+out_str+'.h5',key='PreparedJet_dataframe', mode='w')
    # dump metadata dictionary to pickle:
    print "done.\nStoring metadata dictionary to Pickle..."
    data_dict.update(default_dict)
    with open('PreparedFiles/'+out_str+'__metadata.pkl', 'wb') as output_file:
        pickle.dump(data_dict, output_file)
    print "all done.\n"


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
    parser.add_argument('input_file', type=str, default="inputFiles/flat_ntuple.root",
                        help=help_input_file)
    parser.add_argument('-pt', '--max_pT', type=float, default=300.,
                        choices=[300.,500.],
                        help=help_max_pT)
    parser.add_argument('-c', '--classes', type=str, default="bcu",
                        choices=["bcu","bcut"],
                        help=help_classes)
    parser.add_argument('-train', '--training_fraction', type=float, default=0.8,
                        help=help_training_fraction)
    parser.add_argument('-rw', '--reweighting', default="flat_pT", choices=["b", "bcu", "flat_pT"],
                        help=help_reweighting)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help="increase output verbosity")
    args = parser.parse_args()
    if args.verbose:
        print('Will prepare the sample with the following settings:\n  Input file: {}\n  training fraction: {}\n  max pT: {}\n  Reweighting: {}\n'.format(args.input_file, args.training_fraction, args.max_pT, args.reweighting))
    return args



if __name__ == '__main__':
    _run()
