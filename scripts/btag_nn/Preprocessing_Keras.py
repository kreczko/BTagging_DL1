'''
    Script to train NN for jet flavour identification purpose (b, c and light (and tau) jets): training and evaluation with Keras
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python btagging_nn.py
    Switch Keras backend using: KERAS_BACKEND=tensorflow python btagging_nn.py
'''
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
    # transforms label entries to int32 and array to binary class matrix as required for categorical_crossentropy:
    Y_train = np_utils.to_categorical(y_train.astype(int), nb_classes)
    Y_test = np_utils.to_categorical(y_test.astype(int), nb_classes)
    return X_train, Y_train, X_test, Y_test, weights_training


def load_btagging_data(inFile, validation_set):
    exists_already_check = False
    store_filename = "KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100)))
    if os.path.isfile("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace('.h5','_val%s.h5' % str(int(validation_set*100)))):
        exists_already_check = True
        print "Load stored numpy arrays..."
        X_train = pd.read_hdf(store_filename,"X_train")
        y_train = pd.read_hdf(store_filename,"y_train")
        X_val = pd.read_hdf(store_filename,"X_val")
        y_val = pd.read_hdf(store_filename,"y_val")
        X_test = pd.read_hdf(store_filename,"X_test")
        y_test = pd.read_hdf(store_filename,"y_test")
        sample_weights_training = pd.read_hdf(store_filename,"sample_weights_training")
        sample_weights_validation = pd.read_hdf(store_filename,"sample_weights_validation")
        X_train = X_train.as_matrix()
        X_val = X_val.as_matrix()
        X_test = X_test.as_matrix()
        with open("KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__info.json"),"r") as input_info_file:
            info_dict = json.load(input_info_file)
        nb_features = info_dict.get("nb_inputs")
        nb_classes = info_dict.get("nb_classes")
        print "...done."
    if exists_already_check is False:
        try:
            if not os.path.isfile(inFile):
                print inFile+" does not exist."
                return False
        except IOError as ex:
            print "({})".format(e)

        print "Preprocess data...\n  load data..."
        from btag_nn_inputs import weight_str, append_input_variables, append_kinematic_variables, append_check_variables, flavor_dict, label_dict_Keras

        # read training | testing set separation from HDF5:
        jet_list_training = pd.read_hdf(inFile, "jet_list_training")
        jet_list_testing = pd.read_hdf(inFile, "jet_list_testing")
        jet_list_training = np.asarray(jet_list_training["jet_list_training"])
        jet_list_testing = np.asarray(jet_list_testing["jet_list_testing"])
        df = pd.read_hdf(inFile,key="PreparedJet_dataframe")
        new_defaults_dict = pd.read_hdf(inFile,key='defaults').set_index('variable_name')['default_value'].to_dict()

        print "  done.\n  reduce data..."
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


        print "  done.\n  update labels, calculate normalization offset and scale and split data..."
        # adjusting some variable values (case specific)
        if "bcut" in inFile:
            flavor_dict.update({"tau": 15,})
            label_dict_Keras.update({"tau": 3,})
        for flavor in flavor_dict.keys():
            print "  Set particle-ID "+str(flavor_dict.get(flavor))+" to label number "+str(label_dict_Keras.get(flavor))
            df.update(df["label"].replace(to_replace=flavor_dict.get(flavor), value=label_dict_Keras.get(flavor)), join = "left", overwrite = True) # relabel jets

        print "  done.\n  Store jet lists to HDF5..."
        # split testing set into development and testing sets, then store lists in HDF5 output file:
        jet_list_training, jet_list_validation = np.split(jet_list_training,[int(round(len(jet_list_training)*(1.-validation_set)))])
        pd.DataFrame(data=np.asarray(jet_list_training), columns=['jet_list_training']).to_hdf(store_filename, key='jet_list_training', mode='w')
        pd.DataFrame(data=np.asarray(jet_list_validation), columns=['jet_list_validation']).to_hdf(store_filename, key='jet_list_validation', mode='w')
        pd.DataFrame(data=np.asarray(jet_list_testing), columns=['jet_list_testing']).to_hdf(store_filename, key='jet_list_testing', mode='w')

        print "  done.\n  Restructure DataFrame: fill lists with keras inputs..."
        nb_features = len(feature_array)
        nb_classes = len(classes_array)
        len__jet_list_training = len(jet_list_training)
        len__jet_list_testing = len(jet_list_testing)
        len__jet_list_validation = len(jet_list_validation)

        # retrieve sample weights (1:1 mapping to the training or validation samples)
        sample_weights_training = df[weight_str].loc[jet_list_training]
        sample_weights_validation = df[weight_str].loc[jet_list_validation]

        # remove variables not used in training (like e.g. the event weights)
        df = df.reindex_axis(input_array, axis=1)

        # calculating normalization offset and scale:
        scale = {}
        offset = {}
        for column_name, column_series in df.iteritems():
            offset.update({column_name: -column_series.mean()})
            scale.update({column_name: 1./(column_series).std()})

        info_dict = {
            "nb_inputs": nb_features,
            "nb_classes": nb_classes,
            "nb_training": len__jet_list_training,
            "nb_validation": len__jet_list_validation,
            "nb_testing": len__jet_list_testing,
            "part of training-set for validation-set": validation_set,
            "class_labels": ["u-jet","c-jet","b-jet"],
            "inputs": []
        }

        if "bcut" in inFile:
            info_dict['class_labels'].append('tau-jet')

        # construct actual Keras inputs:
        X_train = []
        X_test = []
        X_val = []

        # training
        print "    Start iterating over ",len__jet_list_training," training set entries..."
        df_train = df.reindex_axis(feature_array, axis=1).loc[jet_list_training]
        X_train = df_train.as_matrix()
        y_train = df.reindex_axis(["label"], axis=1).loc[jet_list_training]["label"].tolist()
        del df_train

        # development
        print "    done.\n    Start iterating over ",len__jet_list_validation," development set entries..."
        df_val = df.reindex_axis(feature_array, axis=1).loc[jet_list_validation]

        X_val = df_val.as_matrix()
        y_val = df.reindex_axis(["label"], axis=1).loc[jet_list_validation]["label"].tolist()
        del df_val

        # testing
        print "    done.\n    Start iterating over ",len__jet_list_testing," testing set entries..."
        df_test = df.reindex_axis(feature_array, axis=1).loc[jet_list_testing]
        X_test = df_test.as_matrix()
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
            X_train.T[feature_itr] = X_train.T[feature_itr] + offset.get(feature_item)
            X_train.T[feature_itr] = X_train.T[feature_itr] * scale.get(feature_item)
            X_val.T[feature_itr] = X_val.T[feature_itr] + offset.get(feature_item)
            X_val.T[feature_itr] = X_val.T[feature_itr] * scale.get(feature_item)
            X_test.T[feature_itr] = X_test.T[feature_itr] + offset.get(feature_item)
            X_test.T[feature_itr] = X_test.T[feature_itr] * scale.get(feature_item)
            if feature_item in new_defaults_dict:
                info_dict["inputs"].append({"name": feature_item, "offset": float(offset.get(feature_item)), "scale": float(scale.get(feature_item)), "default": float(new_defaults_dict.get(feature_item))})
            else:
                info_dict["inputs"].append({"name": feature_item, "offset": float(offset.get(feature_item)), "scale": float(scale.get(feature_item))})

        print "done.\nStore info in JSON and numpy arrays for Keras training in HDF5 format..."
        store = pd.HDFStore(store_filename)
        store.put("X_train", pd.DataFrame(X_train))
        store.put("y_train", pd.DataFrame(y_train))
        store.put("X_val", pd.DataFrame(X_val))
        store.put("y_val", pd.DataFrame(y_val))
        store.put("X_test", pd.DataFrame(X_test))
        store.put("y_test", pd.DataFrame(y_test))
        store.put("sample_weights_training", sample_weights_training)
        store.put("sample_weights_validation", sample_weights_validation)
        store.close()

        info_file_str = "KerasFiles/input/Keras_input__"+inFile.split("/")[1].replace("PreparedSample__","").replace("_ntuple","").replace(".h5","__info.json")

        # add Keras version information:
        #############################################
        os.system("pip show keras &> {}".format(info_file_str.replace(".json","_keras_version.txt")))
        with open(info_file_str.replace(".json","_keras_version.txt"),'r') as f:
            for line in f:
                if "Version" in line:
                    keras_version_str = str(line)
        keras_version_str.strip()
        info_dict.update({"keras_version": keras_version_str.split(':')[1].strip(' ').strip('\n')},)
        os.system("rm {}".format(info_file_str.replace(".json","_keras_version.txt")))
        del keras_version_str
        #############################################

        with open(info_file_str, "w") as info_file:
            json.dump(info_dict, info_file, indent=2, sort_keys=True)
        print "done."
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (sample_weights_training.values, sample_weights_validation.values), (nb_features, nb_classes)
