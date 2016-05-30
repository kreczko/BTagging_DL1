'''
additional functions for the support of using Keras
'''

from __future__ import absolute_import
from __future__ import print_function
import os


def create_output_filestring(input_file, arg, subdirName):
    part_from_input_file__reweighing_part = input_file.split(".h5")[0].split("__")[2]
    part_from_input_file__pTmax_part = input_file.split(".h5")[0].split("__")[1].split("_")[3]
    part_from_input_file__trainFrac_part = input_file.split(".h5")[0].split("__")[1].split("_")[2]
    part_from_input_file = part_from_input_file__trainFrac_part+"_"+part_from_input_file__pTmax_part+"__"+part_from_input_file__reweighing_part
    out_str = subdirName+part_from_input_file
    out_str+="_val"+str(int(arg.validation_set*100))
    print("out_str = ", out_str)
    hist_str = "KerasFiles/"+subdirName+"/Keras_callback_ModelCheckpoint/hist__"+out_str+".h5"
    protocol_str = "KerasFiles/LogFiles/Log__"+out_str+".json"
    architecture_str = "KerasFiles/"+subdirName+"/Keras_output/flavtag_model_architecture__"+out_str+".json"
    weights_str = "KerasFiles/"+subdirName+"/Keras_output/flavtag_model_weights__"+out_str+".h5"
    store_str = "KerasFiles/"+subdirName+"/Keras_output/Keras_output__"+out_str+".h5"
    plot_str = "KerasFiles/"+subdirName+"/Keras_output/plot__"+out_str+"model.png"
    return out_str, hist_str, store_str, protocol_str, architecture_str, weights_str, plot_str


def save_history(hist, hist_filename):
    import numpy as np
    import pandas as pd

    print("save history as:", hist_filename)
    df_history = pd.DataFrame(np.asarray(hist.history.get("loss")), columns=['loss'])
    df_history['acc'] = pd.Series(np.asarray(hist.history.get("acc")), index=df_history.index)
    df_history['val_loss'] = pd.Series(np.asarray(hist.history.get("val_loss")), index=df_history.index)
    df_history['val_acc'] = pd.Series(np.asarray(hist.history.get("val_acc")), index=df_history.index)
    df_history.to_hdf(hist_filename, key='history', mode='w')
