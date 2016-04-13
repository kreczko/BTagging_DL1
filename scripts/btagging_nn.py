#!/usr/bin/env python
'''

    Script to train NN for jet flavour identification purpose (b-, c- and light jets): training and evaluation with Keras

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ./btagging_nn.py

    Switch backend using: KERAS_BACKEND=tensorflow ./btagging_nn.py

'''

from __future__ import absolute_import
from __future__ import print_function
import os, time
import numpy as np
import pandas as pd
import json
import argparse
from keras import backend
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam, Adamax
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils.visualize_util import plot
from keras.models import model_from_json
from btag_nn import load_data_pickle, load_btagging_data_inclFewTestJets, save_history

def _run():
    args = _get_args()
    inFile     = args.input_file
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    np.random.seed(args.seed_nr)  # for reproducibility

    if args.few_testjets[0] is not "0":
        os.system("mkdir -p TestFiles")
        os.system("mkdir -p TestFiles/input")
        os.system("mkdir -p TestFiles/LogFiles")
        os.system("mkdir -p TestFiles/Keras_output")
    else:
        os.system("mkdir -p KerasFiles/")
        os.system("mkdir -p KerasFiles/input/")
        os.system("mkdir -p KerasFiles/LogFiles/")
    protocol_dict = {}

    if args.few_testjets[0] is '0':
        (X_train, y_train), (X_dev, y_dev), (X_test, y_test), sample_weight, nb_features, nb_classes, loading_time  = load_btagging_data_inclFewTestJets(inFile, args.few_testjets[0], args.validation_set)
    elif args.few_testjets[0] is not '0':
        (X_train, y_train), (X_test, y_test), sample_weight, nb_features, nb_classes, loading_time  = load_btagging_data_inclFewTestJets(inFile, args.few_testjets[0], args.validation_set)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    X_train = X_train.reshape(X_train.shape[0], nb_features)
    X_test = X_test.reshape(X_test.shape[0], nb_features)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    sample_weight = sample_weight.astype("float32")
    # transforms label entries to int32 and array to binary class matrix:
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    if args.few_testjets[0] is "0":
        y_dev = np.ravel(y_dev)
        X_dev = X_dev.reshape(X_dev.shape[0], nb_features)
        X_dev = X_dev.astype("float32")
        Y_dev = np_utils.to_categorical(y_dev, nb_classes)
        if args.validation_split > 0.:
            X_test = np.append(X_test, X_dev, axis=0)
            Y_test = np.append(Y_test, Y_dev, axis=0)
            del X_dev, Y_dev
            print("Set sizes:\nTraining set: ", len(X_train), "\nTesting set: ", len(X_test))
            protocol_dict.update({"Number of jets in the validation set": 0,})
        else:
            print("Set sizes:\nTraining set: ", len(X_train), "Development set: ", len(X_dev), "\nTesting set: ", len(X_test))
            protocol_dict.update({"Number of jets in the validation set": len(X_dev),})
        protocol_dict.update({
            "Number of jets in the training set": len(X_train),
            "Number of jets in the testing set": len(X_test),
        })
        for arg in vars(args):
            protocol_dict.update({arg: getattr(args, arg)})

        # model selection (or reloading)
        if args.reload_nn[0]=='':
            from btagging_nn_models import get_model
            
            model, subdir_name = get_model(args.model, nb_features, args.activation_function, args.init_distr, nb_classes)
            subdir_name +="_"+args.optimizer+"_clipn"+str(int(args.clipnorm*100))+"_"+args.objective

            # TODO: break this down - it's currently crashing!
            part_from_inFile__reweighing_part = inFile.split(".h5")[0].split("__")[2]
            part_from_inFile__pTmax_part = inFile.split(".h5")[0].split("__")[1].split("_")[3]
            part_from_inFile__trainFrac_part = inFile.split(".h5")[0].split("__")[1].split("_")[2]
            part_from_inFile = part_from_inFile__trainFrac_part+"_"+part_from_inFile__pTmax_part+"__"+part_from_inFile__reweighing_part

            os.system("mkdir -p KerasFiles/%s/" % (subdir_name))
            os.system("mkdir -p KerasFiles/%s/Keras_output/" % (subdir_name))
            os.system("mkdir -p KerasFiles/%s/Keras_callback_ModelCheckpoint/" % (subdir_name))
        else:
            # reload nn configuration from previous training
            part_from_inFile = "RE__"+args.reload_nn[0].split(".json")[0].split("__")[2]+"__"+args.reload_nn[0].split(".json")[0].split("__")[3].replace("MaxoutLayers","MO").replace("categorical_crossentropy", "cce").replace("DenseLayers","D")
            subdir_name = args.reload_nn[0].split(".json")[0].split('/')[1]
            model = model_from_json(open(args.reload_nn[0]).read())
            model.load_weights(args.reload_nn[1])

        out_str = subdir_name+part_from_inFile+"__trainBS"+str(batch_size)+"_nE"+str(nb_epoch)+"_s"+str(args.seed_nr)
        if args.validation_split > 0.:
            out_str +="_XVal"+str(int(args.validation_split*100))+"_"
        else:
            out_str+="_val"+str(int(args.validation_set*100))

        print("out_str = ", out_str)
        if args.clipnorm!=0.:
            if args.optimizer=="adam":
                model_optimizer=Adam(clipnorm=args.clipnorm)
            elif args.optimizer=="adamax":
                model_optimizer=Adamax(clipnorm=args.clipnorm)
            print(args.objective, args.optimizer, model_optimizer)
            model.compile(loss=args.objective, optimizer=model_optimizer)
        else:
            print(args.objective, args.optimizer)
            model.compile(loss=args.objective, optimizer=args.optimizer)

        # This code is only left in so I know how to use these functions if I ever have to use them
        # Callback: early stopping if loss does not decrease anymore after 10 epochs
        early_stopping = EarlyStopping(monitor="val_loss", patience=10)
        learning_rate_scheduler = LearningRateScheduler(lambda x: 0.1 / (1. + x))

        # Callback: model checkpoint
        model_check_point = ModelCheckpoint("KerasFiles/"+subdir_name+"/Keras_callback_ModelCheckpoint/weights_"+out_str+"__{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="auto")

        if backend._BACKEND is "tensorflow":
            from keras.callbacks import TensorBoard
            os.system("mkdir -p ./KerasFiles/TensorFlow_logs/")
            TensorBoard(log_dir='./KerasFiles/TensorFlow_logs', histogram_freq=2)


        # REMARK: using the Keras split function on the training set to generate a development/validation set would upset the c-fraction in the training!
        start_training = time.time()
        if args.validation_split==0.:
            history = model.fit(X_train, Y_train,
                                batch_size=batch_size, nb_epoch=nb_epoch,
                                callbacks=[model_check_point, early_stopping, learning_rate_scheduler],
                                show_accuracy=True, verbose=1,
                                validation_data=(X_dev, Y_dev),
                                sample_weight=sample_weight) # shuffle=True (default)
        else:
            history = model.fit(X_train, Y_train,
                                batch_size=batch_size, nb_epoch=nb_epoch,
                                callbacks=[model_check_point, early_stopping, learning_rate_scheduler],
                                show_accuracy=True, verbose=1,
                                validation_split=args.validation_split,
                                sample_weight=sample_weight) # shuffle=True (default)
        # store history:
        history_filename = "KerasFiles/"+subdir_name+"/Keras_callback_ModelCheckpoint/hist__"+out_str+".h5"
        save_history(history, history_filename)
        end_training = time.time()
        training_time=end_training-start_training
        print("training time: ", training_time)

    elif args.few_testjets[0]!='0':
        # text dump of info as the Keras-NN gets it
        Keras_input_test_jet_dict = {
            "Keras-style prepared inputs per jet": [],
            "Keras-style prepared labels per jet": []
        }
        for i in X_test:
            Keras_input_test_jet_dict["Keras-style prepared inputs per jet"].append(np.array_str(i))
        for i in Y_test:
            Keras_input_test_jet_dict["Keras-style prepared labels per jet"].append(np.array_str(i))
        with open("TestFiles/input/Keras_inputs__"+args.few_testjets[0]+"_testjets__"+inFile.split("/")[1].replace(".h5",".json"),'a') as tf:
            json.dump(Keras_input_test_jet_dict, tf, indent=2, sort_keys=True)
        # HDF5 dump of info as the Keras-NN gets it
        store = pd.HDFStore("TestFiles/input/Keras_inputs__"+args.few_testjets[0]+"_testjets__"+inFile.split("/")[1])
        store.put("X_test", pd.DataFrame(X_test))
        store.put("Y_test", pd.DataFrame(Y_test))
        store.close()

        Keras_architecture_str = args.few_testjets[1]
        Keras_weight_str = Keras_architecture_str.replace("architecture","weights").replace(".json",".h5")
        model = model_from_json(open(Keras_architecture_str).read())
        model.load_weights(Keras_weight_str)
        subdir_name = Keras_weight_str.split("/")[1]
        out_str = Keras_weight_str.split("weights")[1].split(".")[0]+"_few_testjets"
        protocol_dict.update({"Number of jets in the training set": len(X_train), "Number of jets in the testing set": len(X_test),})
        for arg in vars(args):
            protocol_dict.update({arg: getattr(args, arg),})

    start_evaluation_time = time.time()
    if args.validation_split > 0.:
        score = model.evaluate(X_test, Y_test,
                               show_accuracy=True, verbose=1)
    else:
        score = model.evaluate(X_test, Y_test,
                               show_accuracy=True, verbose=1)
    end_evaluation_time = time.time()
    evaluation_time=end_evaluation_time-start_evaluation_time
    print("evaluation_time = ", evaluation_time)
    print("Classification score:", score[0])
    print("Classification accuracy:", score[1])
    protocol_dict.update({"Classification score": score[0],"Classification accuracy": score[1]})

    start_prediction_time = time.time()
    predictions = model.predict(X_test, batch_size=1, verbose=1) # returns predictions in numpy.array
    #predictions_classes = model.predict_classes(X_test, batch_size=1, verbose=1)
    #predictions_probabilities = model.predict_proba(X_test, batch_size=1, verbose=1)
    end_prediction_time = time.time()
    prediction_time=end_prediction_time-start_prediction_time
    print("prediction_time = ", prediction_time)

    if args.few_testjets[0]!='0':
        store_str = "TestFiles/Keras_output/Keras_output"+out_str+"__"+args.few_testjets[0]+"_testjets.h5"
        output_test_jet_dict = {
            "predictions": np.array_str(predictions),
            #"predictions_classes": np.array_str(predictions_classes),
            #"predictions_probabilities": np.array_str(predictions_probabilities)
        }
        with open("TestFiles/Keras_output/Keras_outputs__"+args.few_testjets[0]+"_testjets__"+inFile.split("/")[1].replace(".h5",".json"),'a') as tf:
            json.dump(output_test_jet_dict, tf, indent=2, sort_keys=True)
        print("\nStored few testjets as:\n  %s (HDF5)\n  %s\n" % (store_str, "TestFiles/Keras_output/Keras_outputs__"+args.few_testjets[0]+"_testjets__"+inFile.split("/")[1].replace(".h5",".json")))

    elif args.few_testjets[0]=='0':
        timing_dict = {
            "Loading time (b-tagging data)": loading_time,
            "Training time": training_time,
            "Evaluation time": evaluation_time,
            "Prediction time": prediction_time
        }
        store_str = "KerasFiles/"+subdir_name+"/Keras_output/Keras_output__"+out_str+".h5"
        # save NN configuration architecture:
        json_string = model.to_json()
        open("KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_architecture__"+out_str+".json", "w").write(json_string)
        # save NN configuration weights:
        model.save_weights("KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_weights__"+out_str+".h5", overwrite=True)
        plot(model, to_file="KerasFiles/"+subdir_name+"/Keras_output/plot__"+out_str+"model.png")

        protocol_dict.update(timing_dict)

    with open("KerasFiles/LogFiles/Log__"+out_str+".json", "w") as protocol_file:
        json.dump(protocol_dict, protocol_file, indent=2, sort_keys=True)

    print("save predictions as ", store_str)
    store = pd.HDFStore(store_str)
    store.put("predictions", pd.DataFrame(predictions))
    store.put("labels", pd.DataFrame(Y_test))
    store.close()


def _get_args():
    help_input_file = "Input file determining the pT and eta ranges as well as the c-fraction in the BG sample (default: %(default)s)."
    help_batch_size = "Batch size: Set the number of jets to look before updating the weights of the NN (default: %(default)s)."
    help_nb_epoch = "Set the number of epochs to train over (default: %(default)s)."
    help_init_distr = "Weight initialization distribution determines how the initial weights are set (default: %(default)s)."
    help_seed_nr = "Seed initialization (default: %(default)s)."
    help_validation_set = "Percentage of the test set to be used as development/validation set (default: %(default)s)."
    help_model = "Architecture of the NN. Due to the current code structure only a few are available for parameter search (default: %(default)s)."
    help_activation_function = "activation function (default: %(default)s)."
    help_optimizer = "optimizer for training (default: %(default)s)."
    help_objective = "objective (or loss) function for training (default: %(default)s)."
    help_few_testjets = "Option to only run over a few jets to get a test output using an earlier trained net (syntax: ['<Number of testjets>', <Keras_architecture_str>], default: %(default)s --> Example could be e.g, type -t '3' 'KerasFiles/DenseLayers_48_48_24_12_6_3_relu/Keras_output/flavtag_model_architecture__DenseLayers_48_48_24_12_6_3_relu__TrainFrac80_cFrac20__trainingbatchsize80_nbepoch20_initlecun_uniform_seed12264_validation20.json')."

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-i", "--input_file", type=str,
                        default="PreparedFiles/PreparedSample__jet_ntuple_TrainFrac80_cFrac20_pTmax300__b_reweighting.h5",
                        help=help_input_file)
    parser.add_argument("-r", "--reload_nn",
                        type=str, nargs='+', default=["",""],
                        help="reload previously trained model, provide architecture (1st argument; JSON) and weights (2nd argument; HDF5) (default: %(default)s).")
    parser.add_argument("-m", "--model",
                        type=str, default="Dense",
                        choices=["Dense"],
                        help=help_model)
    parser.add_argument("-obj", "--objective",
                        type=str, default="categorical_crossentropy",
                        choices=["categorical_crossentropy", "mse"],
                        help=help_objective)
    parser.add_argument("-o", "--optimizer",
                        type=str, default="adam",
                        choices=["adam", "adamax"],
                        help=help_optimizer)
    parser.add_argument("-af", "--activation_function",
                        type=str, default="relu",
                        choices=["relu", "tanh"],
                        help=help_activation_function)
    parser.add_argument("-bs", "--batch_size",
                        type=int, default=80,
                        help=help_batch_size)
    parser.add_argument("-ne", "--nb_epoch",
                        type=int, default=20,
                        help=help_nb_epoch)
    parser.add_argument("-id", "--init_distr",
                        type=str, default="lecun_uniform",
                        help=help_init_distr)
    parser.add_argument("-sn", "--seed_nr",
                        type=int, default=12264,
                        help=help_seed_nr)
    parser.add_argument("-vs", "--validation_set",
                        type=float, default=.3,
                        help=help_validation_set)
    parser.add_argument("-cn", "--clipnorm",
                        type=float, default=0.,
                        help="clipnorm for gradient clipping (default: %(default)s)")
    parser.add_argument("-t", "--few_testjets",
                        type=str, default=["0", ""],
                        nargs='+',
                        help=help_few_testjets)
    parser.add_argument("-vsp", "--validation_split",
                        type=float, default=0.,
                        help="validation split using part of the training set (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="increase output verbosity")
    args = parser.parse_args()
    if args.verbose:
        print("Will train with the following settings:\n  Input file: {}\n  Model: {}\n  Activation function: {}\n  Batch size: {}\n  Number of epochs: {}\n  Initialization distribution of NN weights: {}\n".format(args.input_file, args.model, args.activation_function, args.batch_size, args.nb_epoch, args.init_distr))
    return args


if __name__ == "__main__":
    _run()
