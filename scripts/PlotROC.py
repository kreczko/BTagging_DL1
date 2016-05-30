'''
ROC curve and AUC plotting code
'''

import os, sys, array, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from btag_nn_inputs import jet_eta_str, jet_pt_str, label_dict_Keras

def _run():
    args = _get_args()

    label_dict_Keras['light'] = label_dict_Keras.pop('u')

    # get predictions, labels and metadata
    print "get classification from other b-taggers..."
    df_ST_testing = _check_and_load_standard_taggers(args.input_file, label_dict_Keras, args.signal_flavor, args.rejection_flavor)

    print "done.\narranging information a bit..."
    color_dict = _get_color_dict(args.baseline_tagger, args.cFraction)

    # discs (dictionaries)
    discs = {}
    discs_ratio = {}

    sig_bg_selection_name = args.signal_flavor+"-jet_signal+"+args.rejection_flavor+"-jet_background"
    os.system("mkdir -p Plotting/")
    os.system("mkdir -p Plotting/ROC_curves/")
    os.system("mkdir -p Plotting/ROC_curves/"+sig_bg_selection_name+"/")
    os.system("mkdir -p Plotting/ROC_curves/"+sig_bg_selection_name+"/efficiencies/")


    if not args.roc_file:
        if args.keras_file:
            plotfile_name = "Plotting/ROC_curves/"+sig_bg_selection_name+"/BT_"+args.baseline_tagger+"_"+args.rejection_flavor+"-rej_"+str(args.keras_file.split('/')[3].replace('.h5','.'+args.output_format))
        else:
            plotfile_name = "Plotting/ROC_curves/"+sig_bg_selection_name+"/ROC."+args.output_format
    else:
        plotfile_name = args.roc_file


    # add ROC curve for other ATLAS taggers:
    print "done.\nadding ROC curves for baseline taggers..."
    discs = _calculate_roc_curve(df_ST_testing, label_dict_Keras, args.signal_flavor, args.rejection_flavor, args.baseline_tagger, color_dict, args.bins, args.cFraction, discs, 'ST')

    # Keras output:
    if args.keras_file:
        print "get Keras classification..."
        df_keras = _load_keras_predictions(args.keras_file)
        df_keras = _fix_labels(df_keras, args.signal_flavor, args.rejection_flavor, label_dict_Keras)

        discs = _calculate_roc_curve(df_keras, label_dict_Keras, args.signal_flavor, args.rejection_flavor, args.baseline_tagger, color_dict, args.bins, args.cFraction, discs, 'DL1')


        for tagger, data in discs.iteritems():
            if tagger==args.baseline_tagger:
                baseline_sig_eff = data['efficiency']
                baseline_bkg_rej = data['rejection']
                baseline_reference_arr = np.interp(x=[0.7, 0.77, 0.8], xp=data['efficiency'], fp=data['rejection'])
                break

        for tagger, data in discs.iteritems():
            print "tagger = ",tagger
            DL1_bkg_rej_intrpl = np.interp(x=baseline_sig_eff, xp=data['efficiency'], fp=data['rejection'])
            DL1_bkg_rej_intrpl_ratio = np.divide(DL1_bkg_rej_intrpl, baseline_bkg_rej)
            _add_curve(r''+tagger, color_dict.get(tagger), baseline_sig_eff, DL1_bkg_rej_intrpl_ratio, discs_ratio)


    # plot ROC curves:
    file_name = plotfile_name

    min_eff = 0.6
    max_eff = 1.
    ymax=10**4

    _plot_roc_curve(discs, baseline_tagger=args.baseline_tagger, cFrac=args.cFraction, min_eff = min_eff, max_eff=max_eff, show=args.show_roc, save_filename=file_name, label_xaxis=r" (%s-jets)" % args.signal_flavor, label_yaxis=r"%s-jet rejection" % args.rejection_flavor, logscale=True, ymax=ymax)

    ratio_file_name = plotfile_name.replace("."+args.output_format, "__ratio."+args.output_format)
    ymin=0.6
    ymax=2.5

    _plot_roc_curve(discs_ratio, baseline_tagger=args.baseline_tagger, cFrac=args.cFraction, min_eff = min_eff, max_eff=max_eff, show=args.show_roc, save_filename=ratio_file_name, label_xaxis=r" (%s-jets)" % "b", label_yaxis=r"ratio of %s-jet rejection" % args.rejection_flavor, logscale=False, ymin=ymin, ymax=ymax)

    print "Output:\n  --> saved ROC curve as: ", file_name, "\n  --> saved ROC ratio as: ", ratio_file_name



def _check_and_load_standard_taggers(input_file, label_dict_new, sig_flavor, rej_flavor):
    # read training | testing set separation from HDF5:
    jet_list_testing = np.asarray(pd.read_hdf(input_file, "jet_list_testing")["jet_list_testing"])

    # get data
    df_all = pd.read_hdf(input_file,"PreparedJet_dataframe")
    df_testing = df_all.loc[jet_list_testing]
    df_testing = df_testing.reset_index(drop=True)
    df_testing = _fix_labels(df_testing, sig_flavor, rej_flavor, label_dict_new)
    df_all_list_testing = df_testing.index.tolist()

    return df_testing


def _load_keras_predictions(input_file):
    try:
        if not os.path.isfile(input_file):
            print str(input_file)+' does not exist.'
            return False
    except IOError as ex:
        print '({})'.format(e)
    df = pd.read_hdf(input_file,'predictions')
    labels = pd.read_hdf(input_file,'labels')
    label_arr = np.zeros(len(np.asarray(labels[0])),)
    for node in labels.columns.tolist():
        label_arr += int(node) * np.asarray(labels[node])
    df['label'] = pd.Series(label_arr, index=df.index)
    return df


def _fix_labels(df, sig_flav, rej_flav, label_dict):
    from btag_nn_inputs import flavor_dict

    df.update(df["label"].replace(to_replace=flavor_dict.get("b"), value=label_dict.get("b")), join = "left", overwrite = True) # relabel b-jets
    df.update(df["label"].replace(to_replace=flavor_dict.get("c"), value=label_dict.get("c")), join = "left", overwrite = True) # relabel c-jets
    return df


def _get_color_dict(baseline_tagger, cFraction):
    from matplotlib.pyplot import cm

    color_dict = {}
    color=iter(cm.rainbow(np.linspace(0,1,len(cFraction))))
    for c_fraction in cFraction:
        c=next(color)
        color_dict.update({"DL1c"+str(int(c_fraction*100.)): c,})

    color_dict.update({baseline_tagger: "black"})
    return color_dict


def _add_curve(name, color, eff, rej, dictref):
    """
    This function adds ROC curves to a common dictionary.
    """
    dictref.update(
        {
            name : {
                'efficiency' : eff,
                'rejection' : rej,
                'color' : color
            }
        }
    )


def _add_roc_curve(name, color, eff, rej, sel, discr, discr_bins, dictref):
    """
    This function adds ROC curves to a common dictionary.
    """
    dictref.update(
        {
            name : {
                'efficiency' : eff,
                'rejection' : rej,
                'selection' : sel,
                'discriminant' : discr,
                'discriminant_bins' : discr_bins,
                'color' : color
            }
        }
    )


def _calculate_roc_curve(df, label_dict, signal_flavor, rejection_flavor, baseline_tagger, color_dict, n_bins_roc, fraction, discs_dict, origin):
    """
    This function calculates the cut discriminant for the ROC curve, calculates the ROC curve and adds them to the respective indicated dicstionary.
    """
    df = df[(df['label'] == label_dict.get(rejection_flavor)) | (df['label'] == label_dict.get(signal_flavor))]
    # calculate ROC curves and store them
    if origin=='ST':
        df = df[[baseline_tagger,'label']]

        # perform transformation (MV2 specific):
        df[baseline_tagger] = df[baseline_tagger].add(1.)
        df[baseline_tagger] = df[baseline_tagger].divide(2.)

        # select signal and background for ROC curve:
        label_arr = np.asarray(df['label'])
        sig_bkg_selection = (label_arr== label_dict.get(rejection_flavor)) | (label_arr== label_dict.get(signal_flavor))

        discriminant = np.asarray(df[baseline_tagger])
        discriminant_finite_check_bool_arr = np.isfinite(discriminant)
        label_arr = label_arr[sig_bkg_selection & discriminant_finite_check_bool_arr]
        discriminant = discriminant[sig_bkg_selection & discriminant_finite_check_bool_arr]
        del df
        sig_ind = label_arr == label_dict.get(signal_flavor)
        bkg_ind = label_arr == label_dict.get(rejection_flavor)
        bkg_total = np.sum(label_arr == label_dict.get(rejection_flavor))
        sig_total = np.sum(label_arr == label_dict.get(signal_flavor))
        discriminant_bins = np.linspace(np.min(discriminant), np.max(discriminant), n_bins_roc)
        sig, _ = np.histogram(discriminant[sig_ind], discriminant_bins)
        bkg, _ = np.histogram(discriminant[bkg_ind], discriminant_bins)
        sig_eff = np.add.accumulate(sig[::-1]) / float(sig_total)
        bkg_rej = 1 / (np.add.accumulate(bkg[::-1]) / float(bkg_total))
        _add_roc_curve(r'{}'.format(baseline_tagger), color_dict.get(baseline_tagger), sig_eff, bkg_rej, sig_bkg_selection & discriminant_finite_check_bool_arr, discriminant, discriminant_bins, discs_dict)
    elif origin=='DL1':
        keras_dict = {"b": np.asarray(df[label_dict.get("b")]),
                      "c":np.asarray(df[label_dict.get("c")]),
                      "light":np.asarray(df[label_dict.get("light")]),
                      "label":np.asarray(df['label'])
        }
        del df
        # select signal and background for ROC curve:
        sig_bkg_selection = (keras_dict.get('label') == label_dict.get(rejection_flavor)) | (keras_dict.get('label') == label_dict.get(signal_flavor))
        # calculate ROC curve for different c- and light-jet fractions (in case of b-jet tagging)
        for frac in fraction:
            if signal_flavor=="b":
                discriminant = np.log(keras_dict.get("b") / ((keras_dict.get("c")*frac + keras_dict.get("light")*(1.-frac))))
            discriminant_finite_check_bool_arr = np.isfinite(discriminant)
            label_arr = keras_dict.get('label')[sig_bkg_selection & discriminant_finite_check_bool_arr]
            discriminant = discriminant[sig_bkg_selection & discriminant_finite_check_bool_arr]
            sig_ind = label_arr == label_dict.get(signal_flavor)
            bkg_ind = label_arr == label_dict.get(rejection_flavor)
            bkg_total = np.sum(label_arr == label_dict.get(rejection_flavor))
            sig_total = np.sum(label_arr == label_dict.get(signal_flavor))
            discriminant_bins = np.linspace(np.min(discriminant), np.max(discriminant), n_bins_roc)
            sig, _ = np.histogram(discriminant[sig_ind], discriminant_bins)
            bkg, _ = np.histogram(discriminant[bkg_ind], discriminant_bins)
            sig_eff = np.add.accumulate(sig[::-1]) / float(sig_total)
            bkg_rej = 1 / (np.add.accumulate(bkg[::-1]) / float(bkg_total))
            _add_roc_curve(r'DL1c'+str(int(frac*100.)), color_dict.get("DL1c"+str(int(frac*100.))), sig_eff, bkg_rej, sig_bkg_selection & discriminant_finite_check_bool_arr, discriminant, discriminant_bins, discs_dict)
    return discs_dict


def _plot_roc_curve(curves, cFrac, baseline_tagger="mv2c20",
                    show=False, save_filename="ROC.pdf", label_xaxis="", label_yaxis="",
                    min_eff = 0.6, max_eff = 1.,
                    logscale=False,
                    ymin=1., ymax=10**4):
    fig = plt.figure(figsize=(11.69, 8.210), dpi=100)
    ax = fig.add_subplot(111)
    plt.xlim(min_eff,max_eff)
    plt.grid(b = True, which = 'minor')
    plt.grid(b = True, which = 'major')
    mv2=True
    list_tagger = [baseline_tagger]
    for c in cFrac:
        list_tagger.append("DL1c"+str(int(c*100.)))

    for t in list_tagger:
        for tagger, tagger_data in curves.iteritems():
            if mv2 and tagger==baseline_tagger:
                if ("ratio" in label_yaxis) and len(cFrac)==2:
                    plt.plot(tagger_data['efficiency'], tagger_data['rejection'], '-', color=tagger_data['color'], linewidth=1.8)
                else:
                    plt.plot(tagger_data['efficiency'], tagger_data['rejection'], '-', label = r''+tagger, color=tagger_data['color'], linewidth=1.8)
                mv2=False
                break
            elif t==tagger:
                plt.plot(tagger_data['efficiency'], tagger_data['rejection'], '-', label = r''+tagger, color=tagger_data['color'], linewidth=1.8)
                break
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)
    plt.ylim(ymin,ymax)
    if logscale:
        ax.set_yscale('log')
    ax.set_xlabel(r"$\epsilon_{\mathrm{signal}}$"+label_xaxis)
    ax.set_ylabel(r""+label_yaxis)
    if "ratio" not in label_yaxis:
        plt.legend(prop={'size':22}, frameon=False)
    elif "ratio" in label_yaxis:
        plt.legend(prop={'size':20}, frameon=False, loc=0)
    if show:
        plt.show()
    fig.savefig(save_filename)




def _get_args():
    # TODO: make exclusive groups
    help_input_file = "Input file containing the standard ATLAS taggers and its name defines the associated metadatafile containing information about the pT and eta ranges as well as the c-fraction in the BG sample (default: %(default)s)."
    help_roc_file = "ROC plot file name (default: %(default)s; will be replaced with 'Plotting/ROC_curves/<input_file>.pdf')."
    help_rejection_flavor = "rejection flavor (default: %(default)s)."
    help_signal_flavor = "signal flavor (default: %(default)s)."
    help_keras_file = 'Keras output file which contains predictions (default: %(default)s).'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument('-in', '--input_file',
                        default='PreparedFiles/PreparedSample__V47full_Akt4EMTo_bcujets_pTmax300GeV_TrainFrac85__b_reweighting.h5',
                        help=help_input_file)
    parser.add_argument('-roc', '--roc_file',
                        default='',
                        help=help_roc_file)
    parser.add_argument('-of', '--output_format',
                        default='pdf',
                        choices=['pdf','png'],
                        help="output format of plots (default: %(default)s')")
    parser.add_argument('-rej', '--rejection_flavor',
                        default='light',
                        choices=['light','c'],
                        help=help_rejection_flavor)
    parser.add_argument('-sig', '--signal_flavor',
                        default="b",
                        choices=["b"],
                        help=help_signal_flavor)
    parser.add_argument('-k', '--keras_file',
                        default='KerasFiles/D_48_48_24_12_6_3SReLU_adam_clipn2_categorical_crossentropy/Keras_output/Keras_output__D_48_48_24_12_6_3SReLU_adam_clipn2_categorical_crossentropy__TrainFrac85_pTmax300__b_reweighting_theano__lr1_trainBS40_nE50_s1337_val20.h5',
                        help=help_keras_file)
    parser.add_argument('-cFrac', '--cFraction',
                        type=float, nargs='+',
                        default=[0., 0.15, 1.],
                        help="Fraction of c-jets in background sample (default: %(default)s').")
    parser.add_argument('-b', '--bins',
                        type=int,
                        default=5000,
                        help="Number of bins in ROC curve plot (default: %(default)s').")
    parser.add_argument('-base', '--baseline_tagger',
                        default='mv2c20',
                        help='baseline tagger to be considered next to the one trained with Keras (default: %(default)s).')
    parser.add_argument('-s', '--show_roc', action='store_true',
                        help='show ROC curves')
    args = parser.parse_args()
    return args



############################ DONE ##########################

if __name__ == '__main__':
    _run()
