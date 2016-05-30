"""
Script to plot the loss and accuracy per epoch from the training history.
"""
import argparse

def _run():
    import pandas as pd

    args = _get_args()

    df_hist = pd.read_hdf(args.input_file, key='history')

    name = args.input_file.replace('.h5','__loss.'+args.output_format)
    _plot_history(df_hist, 'loss', name, args.show)
    name = args.input_file.replace('.h5','__acc.'+args.output_format)
    _plot_history(df_hist, 'acc', name, args.show)


def _plot_history(df, hist_option, name, show_option):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    columns = []
    for column in df.columns.tolist():
        if hist_option in column:
            columns.append(column)

    color_dict = {
        "loss": 'dodgerblue',
        "val_loss": 'lime',
        "acc": 'darkorange',
        "val_acc": 'mediumvioletred'
    }
    print df
    for c in columns:
        plt.plot(df.index.tolist(), np.asarray(df[c]), '-', label = r''+c, color=color_dict.get(c), linewidth=1.8)
    plt.legend(prop={'size':22}, frameon=False)
    plt.xlabel("number of epochs")
    plt.ylabel(hist_option)
    plt.grid(True)
    plt.savefig(name)
    print "--> save ", hist_option, " as: ", name
    if show_option:
        plt.show()


def _get_args():
    help_input_file = "Input file containing the loss and accuracy history (default: %(default)s)."

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-in", "--input_file", type=str,
                        default="KerasFiles/D_48_24_9_3relu_adam_clipn0_categorical_crossentropy/Keras_output/hist__D_48_24_9_3relu_adam_clipn0_categorical_crossentropybcujets_pTmax300GeV__b_reweighting_theano__lr1_trainBS500_nE1_s12264_val30.h5",
                        help=help_input_file)
    parser.add_argument("-of", "--output_format", type=str,
                        default="png",
                        choices=["pdf", "png"],
                        help="output file format (default: %(default)s).")
    parser.add_argument("-s", "--show", action="store_true",
                        help="show final plots")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
        _run()
