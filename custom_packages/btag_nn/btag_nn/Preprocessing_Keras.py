'''

    Script to train NN for jet flavour identification purpose (b-, c- and light jets): training and evaluation with Keras

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python btagging_nn.py

    Switch backend using: KERAS_BACKEND=tensorflow python btagging_nn.py

'''

from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import json


def transform_for_Keras(X_train, y_train, X_test, y_test, weights_training, nb_features, nb_classes):
    from keras.utils import np_utils

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    X_train = X_train.reshape(X_train.shape[0], nb_features)
    X_test = X_test.reshape(X_test.shape[0], nb_features)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    weights_training = weights_training.astype("float32")
    # transforms label entries to int32 and array to binary class matrix:
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test, weights_training


def load_btagging_data(inFile, validation_set):
    exist_check = False # ugly, yes, but works (might be replaced)
    if os.path.isfile("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100)))):
        exist_check = True
        print("Load stored numpy arrays...")
        X_train = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"X_train")
        y_train = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"y_train")
        X_val = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"X_val")
        y_val = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"y_val")
        X_test = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"X_test")
        y_test = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"y_test")
        sample_weight_training = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"sample_weight_training")
        sample_weight_validation = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"sample_weight_validation")
        X_train = X_train.as_matrix()
        X_val = X_val.as_matrix()
        X_test = X_test.as_matrix()
        with open("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__metadata.json"),"r") as input_metadata_file:
            info_dict = json.load(input_metadata_file)
        nb_features = info_dict["general_info"]["nb_inputs"]
        nb_classes = info_dict["general_info"]["nb_classes"]
        print("...done.")
    if exist_check is False:
        try:
            if not os.path.isfile(inFile):
                print(inFile+" does not exist.")
                return False
        except IOError as ex:
            print("({})".format(e))

        print("Preprocess data...\n  load data...")
        from btag_nn_inputs import get_weight_str, append_input_variables, append_kinematic_variables, append_check_variables
        weight_str = get_weight_str()

        # read training | testing set separation from HDF5:
        jet_list_training = pd.read_hdf(inFile.split("__")[0]+"__"+inFile.split("__")[1]+"__jet_lists.h5", "PreparedJet__jet_list_training")
        jet_list_testing = pd.read_hdf(inFile.split("__")[0]+"__"+inFile.split("__")[1]+"__jet_lists.h5", "PreparedJet__jet_list_testing")
        jet_list_training = np.asarray(jet_list_training["jet_list_training"])
        jet_list_testing = np.asarray(jet_list_testing["jet_list_testing"])
        metadata =  load_data_pickle(inFile.replace(".h5","__metadata.pkl"))
        # for speed-up in case of storing only few jets (should not be performed on the training set as this would disturb the normalization of the inputs (i.e. offset and scale))
        df = pd.read_hdf(inFile,"PreparedJet_dataframe")

        print("  done.\n  reduce data...")
        # reduce the DataFrame to the training variables
        feature_array = append_kinematic_variables([])
        intermediate_tagger_variables = append_input_variables([])
        intermediate_tagger_variables = append_check_variables(intermediate_tagger_variables)
        for var in intermediate_tagger_variables:
            feature_array.append(var)

        classes_array = ["jetflav_bottom", "jetflav_charm", "jetflav_light"]
        if "bcut" in inFile:
            classes_array.append("jetflav_tau")
        input_array = [x for x in feature_array]
        input_array.append("label")


        print("  done.\n  update labels, calculate normalization offset and scale and split data...")
        # adjusting some variable values (case specific)
        label_dict_old = {
            "b": 5,
            "c": 4,
            "u": 0
        }
        label_dict_new = {
            "b": 2,
            "c": 1,
            "u": 0
        }
        if "bcut" in inFile:
            label_dict_old.update({"tau": 15,})
            label_dict_new.update({"tau": 3,})
        for flavor in label_dict_old.keys():
            print("  Set particle-ID "+str(label_dict_old.get(flavor))+" to label number "+str(label_dict_new.get(flavor)))
            df.update(df["label"].replace(to_replace=label_dict_old.get(flavor), value=label_dict_new.get(flavor)), join = "left", overwrite = True) # relabel jets


        # split testing set into development and testing sets:
        jet_list_testing, jet_list_validation = np.split(jet_list_testing,[int(round(len(jet_list_testing)*(1.-validation_set)))])
        if not os.path.isfile("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100)))):
            print("  done.\n  Store jet lists to HDF5...")
            df__jet_list_training = pd.DataFrame(data=np.asarray(jet_list_training), columns=['jet_list_training'])
            df__jet_list_validation = pd.DataFrame(data=np.asarray(jet_list_validation), columns=['jet_list_validation'])
            df__jet_list_testing = pd.DataFrame(data=np.asarray(jet_list_testing), columns=['jet_list_testing'])
            df__jet_list_training.to_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100))), key='PreparedJet__jet_list_training', mode='w')
            df__jet_list_validation.to_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100))), key='PreparedJet__jet_list_validation', mode='a')
            df__jet_list_testing.to_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100))), key='PreparedJet__jet_list_testing', mode='a')

        print("  done.\n  Restructure DataFrame: fill lists with keras inputs...")
        nb_features = len(feature_array)
        nb_classes = len(classes_array)
        len__jet_list_training = len(jet_list_training)
        len__jet_list_testing = len(jet_list_testing)
        len__jet_list_validation = len(jet_list_validation)

        # retrieve sample weights (1:1 mapping to the training or validation samples)
        sample_weight_training = df[weight_str].loc[jet_list_training]
        sample_weight_validation = df[weight_str].loc[jet_list_validation]
        
        # remove variables not used in training (like e.g. the event weights)
        df = df.reindex_axis(input_array, axis=1)
        
        # calculating normalization offset and scale:
        scale_pandas = {}
        offset_pandas = {}
        for column_name, column_series in df.iteritems():
            offset_pandas.update({column_name: -column_series.mean()})
            scale_pandas.update({column_name: 1./(column_series).std()})
        
        info_dict = {
            "general_info":{
                "nb_inputs": nb_features,
                "nb_classes": nb_classes,
                "nb_training": len__jet_list_training,
                "nb_validation": len__jet_list_validation,
                "nb_testing": len__jet_list_testing,
                "percentage of test-set for validation-set": validation_set,
            "class_labels": ["u-jet","c-jet","b-jet"],
            },
            "inputs": []
        }

        if "bcut" in inFile:
            info_dict.update({"class_labels": ["u-jet","c-jet","b-jet", "tau-jet"],})

        # construct actual Keras inputs:
        X_train = []
        X_test = []
        X_val = []
        # append input to python lists:

        # training
        print("    Start iterating over ",len__jet_list_training," training set entries...")
        df_train = df.reindex_axis(feature_array, axis=1).loc[jet_list_training]
        for row in df_train.iterrows():
            index, data = row
            X_train.append(data.tolist())
        y_train = df.reindex_axis(["label"], axis=1).loc[jet_list_training]["label"].tolist()
        del df_train

        # development
        print("    done.\n    Start iterating over ",len__jet_list_validation," development set entries...")
        df_val = df.reindex_axis(feature_array, axis=1).loc[jet_list_validation]
        for row in df_val.iterrows():
            index, data = row
            X_val.append(data.tolist())
        y_val = df.reindex_axis(["label"], axis=1).loc[jet_list_validation]["label"].tolist()
        del df_val

        # testing
        print("    done.\n    Start iterating over ",len__jet_list_testing," testing set entries...")
        df_test = df.reindex_axis(feature_array, axis=1).loc[jet_list_testing]
        for row in df_test.iterrows():
            index, data = row
            X_test.append(data.tolist())
        y_test = df.reindex_axis(["label"], axis=1).loc[jet_list_testing]["label"].tolist()
        del df_test

        # convert lists into numpy arrays:
        X_train = np.asarray(X_train,dtype=np.float32)
        y_train = np.asarray(y_train,dtype=np.float32)
        X_val = np.asarray(X_val,dtype=np.float32)
        y_val = np.asarray(y_val,dtype=np.float32)
        X_test = np.asarray(X_test,dtype=np.float32)
        y_test = np.asarray(y_test,dtype=np.float32)

        for feature_itr, feature_item in enumerate(feature_array):
            X_train.T[feature_itr] = X_train.T[feature_itr] + offset_pandas.get(feature_item)
            X_train.T[feature_itr] = X_train.T[feature_itr] * scale_pandas.get(feature_item)
            X_val.T[feature_itr] = X_val.T[feature_itr] + offset_pandas.get(feature_item)
            X_val.T[feature_itr] = X_val.T[feature_itr] * scale_pandas.get(feature_item)
            X_test.T[feature_itr] = X_test.T[feature_itr] + offset_pandas.get(feature_item)
            X_test.T[feature_itr] = X_test.T[feature_itr] * scale_pandas.get(feature_item)
        if feature_item in metadata:
            info_dict["inputs"].append({"name": feature_item, "offset": float(offset_pandas.get(feature_item)), "scale": float(scale_pandas.get(feature_item)), "default": float(metadata[feature_item])})
        else:
            info_dict["inputs"].append({"name": feature_item, "offset": float(offset_pandas.get(feature_item)), "scale": float(scale_pandas.get(feature_item))})

        print("done.\nStore metadaa in JSON and numpy arrays for Keras training in HDF5 format...")
        store = pd.HDFStore("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))))
        store.put("X_train", pd.DataFrame(X_train))
        store.put("y_train", pd.DataFrame(y_train))
        store.put("X_val", pd.DataFrame(X_val))
        store.put("y_val", pd.DataFrame(y_val))
        store.put("X_test", pd.DataFrame(X_test))
        store.put("y_test", pd.DataFrame(y_test))
        store.put("sample_weight_training", sample_weight_training)
        store.put("sample_weight_validation", sample_weight_validation)
        store.close()

        meta_info_file_str = "KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__metadata.json")

        with open(meta_info_file_str, "w") as meta_info_file:
            json.dump(info_dict, meta_info_file, indent=2, sort_keys=True)
        print("done.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (sample_weight_training.values, sample_weight_validation.values), (nb_features, nb_classes)


def load_btagging_data_inclFewTestJets(inFile, few_testjets, validation_set):
    print(inFile, few_testjets, validation_set)
    import time
    start_load_btagging_data = time.time()
    exist_check = False # ugly, yes, but works (might be replaced)
    if few_testjets is not "0" and os.path.isfile("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","")):
        exist_check = True
        print("Load stored numpy arrays of few test-jets...")
        X_train = pd.read_hdf("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple",""),"X_train")
        y_train = pd.read_hdf("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple",""),"y_train")
        X_test = pd.read_hdf("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple",""),"X_test")
        y_test = pd.read_hdf("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple",""),"y_test")
        sample_weight_training = pd.read_hdf("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple",""),"sample_weight_training")
        X_train = X_train.as_matrix()
        X_test = X_test.as_matrix()
        with open("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__metadata.json"),"r") as input_metadata_file:
            info_dict = json.load(input_metadata_file)
        nb_features = info_dict["general_info"]["nb_inputs"]
        nb_classes = info_dict["general_info"]["nb_classes"]
        print("...done.")
    elif few_testjets is "0" and os.path.isfile("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100)))):
        exist_check = True
        print("Load stored numpy arrays...")
        X_train = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"X_train")
        y_train = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"y_train")
        X_val = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"X_val")
        y_val = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"y_val")
        X_test = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"X_test")
        y_test = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"y_test")
        sample_weight_training = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"sample_weight_training")
        sample_weight_validation = pd.read_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))),"sample_weight_validation")
        X_train = X_train.as_matrix()
        X_val = X_val.as_matrix()
        X_test = X_test.as_matrix()
        with open("KerasFiles/input/Keras_input_"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__metadata.json"),"r") as input_metadata_file:
            info_dict = json.load(input_metadata_file)
        nb_features = info_dict["general_info"]["nb_inputs"]
        nb_classes = info_dict["general_info"]["nb_classes"]
        print("...done.")
    if exist_check is False:
        try:
            if not os.path.isfile(inFile):
                print(inFile+" does not exist.")
                return False
        except IOError as ex:
            print("({})".format(e))

        print("Preprocess data...\n  load data...")
        from btag_nn_inputs import get_weight_str, append_input_variables, append_kinematic_variables, append_check_variables
        weight_str = get_weight_str()

        # read training | testing set separation from HDF5:
        jet_list_training = pd.read_hdf(inFile.split("__")[0]+"__"+inFile.split("__")[1]+"__jet_lists.h5", "PreparedJet__jet_list_training")
        jet_list_testing = pd.read_hdf(inFile.split("__")[0]+"__"+inFile.split("__")[1]+"__jet_lists.h5", "PreparedJet__jet_list_testing")
        jet_list_training = np.asarray(jet_list_training["jet_list_training"])
        jet_list_testing = np.asarray(jet_list_testing["jet_list_testing"])
        metadata =  load_data_pickle(inFile.replace(".h5","__metadata.pkl"))
        # for speed-up in case of storing only few jets (should not be performed on the training set as this would disturb the normalization of the inputs (i.e. offset and scale))
        if few_testjets is not "0":
            jet_list_testing=jet_list_testing[:int(few_testjets)]
        df = pd.read_hdf(inFile,"PreparedJet_dataframe")

        print("  done.\n  reduce data...")
        # reduce the DataFrame to the training variables
        feature_array = append_kinematic_variables([])
        intermediate_tagger_variables = append_input_variables([])
        intermediate_tagger_variables = append_check_variables(intermediate_tagger_variables)
        for var in intermediate_tagger_variables:
            feature_array.append(var)

        classes_array = ["jetflav_bottom","jetflav_charm","jetflav_light"]
        if "bcut" in inFile:
            classes_array.append("jetflav_tau")
        input_array = [x for x in feature_array]
        input_array.append("label")

        print("  done.\n  update labels, calculate normalization offset and scale and split data...")
        # adjusting some variable values (case specific)
        label_dict_old = {
            "b": 5,
            "c": 4,
            "u": 0
        }
        label_dict_new = {
            "b": 2,
            "c": 1,
            "u": 0
        }
        if "bcut" in inFile:
            label_dict_old.update({"tau": 15,})
            label_dict_new.update({"tau": 3,})
        for flavor in label_dict_old.keys():
            print("    set particle-ID "+str(label_dict_old.get(flavor))+" to label number "+str(label_dict_new.get(flavor)))
            df.update(df["label"].replace(to_replace=label_dict_old.get(flavor), value=label_dict_new.get(flavor)), join = "left", overwrite = True) # relabel b-jets

        # split testing set into development and testing sets:
        if few_testjets is not "0":
            jet_list_testing = jet_list_testing[:int(few_testjets)*2]
            jet_list_testing, jet_list_validation = np.split(jet_list_testing,[intint(few_testjets)])
        else:
            jet_list_testing, jet_list_validation = np.split(jet_list_testing,[int(round(len(jet_list_testing)*(1.-validation_set)))])
            if not os.path.isfile("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100)))):
                print("    done.\n    Store jet lists to HDF5...")
                df__jet_list_training = pd.DataFrame(data=np.asarray(jet_list_training), columns=['jet_list_training'])
                df__jet_list_validation = pd.DataFrame(data=np.asarray(jet_list_validation), columns=['jet_list_validation'])
                df__jet_list_testing = pd.DataFrame(data=np.asarray(jet_list_testing), columns=['jet_list_testing'])
                df__jet_list_training.to_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100))), key='PreparedJet__jet_list_training', mode='w')
                df__jet_list_validation.to_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100))), key='PreparedJet__jet_list_validation', mode='a')
                df__jet_list_testing.to_hdf("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s__jet_lists.h5' % str(int(validation_set*100))), key='PreparedJet__jet_list_testing', mode='a')

        print("    done.\n  restructure DataFrame: fill lists with keras inputs...")
        nb_features = len(feature_array)
        nb_classes = len(classes_array)
        len__jet_list_training = len(jet_list_training)
        len__jet_list_testing = len(jet_list_testing)
        len__jet_list_validation = len(jet_list_validation)

        # retrieve sample weights (1:1 mapping to the training or validation samples)
        sample_weight_training = df[weight_str].loc[jet_list_training]
        sample_weight_validation = df[weight_str].loc[jet_list_validation]

        # remove variables not used in training (like e.g. the event weights)
        df = df.reindex_axis(input_array, axis=1)
        
        # calculating normalization offset and scale:
        scale_pandas = {}
        offset_pandas = {}
        for column_name, column_series in df.iteritems():
            offset_pandas.update({column_name: -column_series.mean()})
            scale_pandas.update({column_name: 1./(column_series).std()})

        info_dict = {
            "general_info":{
                "nb_inputs": nb_features,
                "nb_classes": nb_classes,
                "nb_training": len__jet_list_training,
                "nb_validation": len__jet_list_validation,
                "nb_testing": len__jet_list_testing,
                "percentage of test-set for validation-set": validation_set,
                "class_labels": ["u-jet","c-jet","b-jet"],
                },
            "inputs": []
        }

        if "bcut" in inFile:
            info_dict.update({"class_labels": ["u-jet","c-jet","b-jet", "tau-jet"],})

        # construct actual Keras inputs:
        X_train = []
        X_test = []
        X_val = []
        # append input to python lists:

        # training
        print("    Start iterating over ",len__jet_list_training," training set entries...")
        # discarding unneccesssary information, speed-up for writing lists (to be converted into numpy arrays); additional benefit: the sequence in X_t* will be the same as for feature_array (see "PreparedSample__jet_ntuple__metadata_Keras.pkl"):
        df_train = df.reindex_axis(feature_array, axis=1).loc[jet_list_training]
        for row in df_train.iterrows():
            index, data = row
            X_train.append(data.tolist())
        y_train = df.reindex_axis(["label"], axis=1).loc[jet_list_training]["label"].tolist()
        del df_train

        # development
        print("    done.\n    Start iterating over ",len__jet_list_validation," development set entries...")
        df_val = df.reindex_axis(feature_array, axis=1).loc[jet_list_validation]
        for row in df_val.iterrows():
            index, data = row
            X_val.append(data.tolist())
        y_val = df.reindex_axis(["label"], axis=1).loc[jet_list_validation]["label"].tolist()
        del df_val

        # testing
        print("    done.\n    Start iterating over ",len__jet_list_testing," testing set entries...")
        df_test = df.reindex_axis(feature_array, axis=1).loc[jet_list_testing]
        for row in df_test.iterrows():
            index, data = row
            X_test.append(data.tolist())
        y_test = df.reindex_axis(["label"], axis=1).loc[jet_list_testing]["label"].tolist()
        del df_test

        # convert lists into numpy arrays:
        X_train = np.asarray(X_train,dtype=np.float32)
        y_train = np.asarray(y_train,dtype=np.float32)
        X_val = np.asarray(X_val,dtype=np.float32)
        y_val = np.asarray(y_val,dtype=np.float32)
        X_test = np.asarray(X_test,dtype=np.float32)
        y_test = np.asarray(y_test,dtype=np.float32)

        if few_testjets is not "0":
            store = pd.HDFStore("TestFiles/input/Keras_input_test_init__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple",""))
            store.put("X_test", pd.DataFrame(X_test))
            store.put("y_test", pd.DataFrame(y_test))
            store.close()
            input_test_jet_dict = {
                "Not-normalized inputs per jet": [],
                "Normalized inputs per jet": [],
                "Jet labels": [],
            }
            for i in X_test:
                input_test_jet_dict["Not-normalized inputs per jet"].append(np.array_str(i))
            for i in y_test:
                input_test_jet_dict["Jet labels"].append(np.array_str(i))

        for feature_itr, feature_item in enumerate(feature_array):
            X_train.T[feature_itr] = X_train.T[feature_itr] + offset_pandas.get(feature_item)
            X_train.T[feature_itr] = X_train.T[feature_itr] * scale_pandas.get(feature_item)
            X_val.T[feature_itr] = X_val.T[feature_itr] + offset_pandas.get(feature_item)
            X_val.T[feature_itr] = X_val.T[feature_itr] * scale_pandas.get(feature_item)
            X_test.T[feature_itr] = X_test.T[feature_itr] + offset_pandas.get(feature_item)
            X_test.T[feature_itr] = X_test.T[feature_itr] * scale_pandas.get(feature_item)
            if feature_item in metadata:
                info_dict["inputs"].append({"name": feature_item, "offset": float(offset_pandas.get(feature_item)), "scale": float(scale_pandas.get(feature_item)), "default": float(metadata[feature_item])})
        else:
            info_dict["inputs"].append({"name": feature_item, "offset": float(offset_pandas.get(feature_item)), "scale": float(scale_pandas.get(feature_item))})

        print("  ...done.\n  Store metadata in JSON and numpy arrays for Keras training in HDF5 format...")
        if few_testjets is not "0":
            for i in X_test:
                input_test_jet_dict["Normalized inputs per jet"].append(np.array_str(i))
            # store inputs (not normalized, normalized and labels) in text format; list placement indicates jet number:
            with open("TestFiles/input/Keras_inputs__"+few_testjets+"_testjets__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5",".json"),'w') as tf:
                json.dump(input_test_jet_dict, tf, indent=2, sort_keys=True)
            # slim down file size in case of a simple test dump run
            X_train = X_train[:int(few_testjets)]
            y_train = y_train[:int(few_testjets)]
            store = pd.HDFStore("TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple",""))
        else:
            store = pd.HDFStore("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100))))
        store.put("X_train", pd.DataFrame(X_train))
        store.put("y_train", pd.DataFrame(y_train))
        store.put("X_val", pd.DataFrame(X_val))
        store.put("y_val", pd.DataFrame(y_val))
        store.put("X_test", pd.DataFrame(X_test))
        store.put("y_test", pd.DataFrame(y_test))
        store.put("sample_weight_training", sample_weight_training)
        store.put("sample_weight_validation", sample_weight_validation)
        store.close()

        if few_testjets is not "0":
            meta_info_file_str = "TestFiles/input/Keras_input_test__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__metadata.json")
        else:
            meta_info_file_str = "KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__metadata.json")

        with open(meta_info_file_str, "w") as meta_info_file:
            json.dump(info_dict, meta_info_file, indent=2, sort_keys=True)

        print("...done.")
    end_load_btagging_data = time.time()
    time_load_btagging_data = end_load_btagging_data - start_load_btagging_data
    print("time_load_btagging_data = ", time_load_btagging_data)
    if few_testjets is not "0":
        return (X_train, y_train), (X_test, y_test), sample_weight_training.values, nb_features, nb_classes, time_load_btagging_data
    elif few_testjets is "0":
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), (sample_weight_training.values, sample_weight_validation.values), nb_features, nb_classes, time_load_btagging_data



def load_data_pickle(file_name):
    import pickle
    try:
        with open(file_name, "r") as hkl_file:
            data = pickle.load(hkl_file)
            hkl_file.close()
            return data
    except IOError as e:
        print("({})".format(e))
