#!/usr/bin/env python
'''

    Script to train NN for jet flavour identification purpose (b-, c- and light jets): training and evaluation with Keras

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ./btagging_nn.py

    Switch backend using: KERAS_BACKEND=tensorflow ./btagging_nn.py

'''
from __future__ import print_function
import numpy as np
import pandas as pd
import os, time, json, argparse
from keras import backend
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler
from keras.utils.visualize_util import plot
from keras.models import model_from_json

from btag_nn import load_btagging_data, save_history, transform_for_Keras
from btagging_nn_models import get_model

def _run():
    args = _get_args()
    np.random.seed(args.seed_nr)  # for reproducibility

    os.system("mkdir -p KerasFiles/")
    os.system("mkdir -p KerasFiles/input/")
    os.system("mkdir -p KerasFiles/LogFiles/")
    protocol_dict = {}

    loading_time = time.time()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (sample_weights_training, sample_weights_validation), (nb_features, nb_classes)  = load_btagging_data(args.input_file, args.validation_set)
    loading_time = time.time() - loading_time

    X_train, Y_train, X_test, Y_test, sample_weights_training = transform_for_Keras(X_train, y_train, X_test, y_test, sample_weights_training, nb_features, nb_classes)

    y_val = np.ravel(y_val)
    X_val = X_val.reshape(X_val.shape[0], nb_features)
    X_val = X_val.astype('float32')
    Y_val = np_utils.to_categorical(y_val.astype(int), nb_classes)
    sample_weights_validation = sample_weights_validation.astype("float32")
    if args.validation_split > 0.:
        X_train = np.append(X_train, X_val, axis=0)
        Y_train = np.append(Y_train, Y_val, axis=0)
        sample_weights_training = np.append(sample_weights_training, sample_weights_validation, axis=0)
        del X_val, Y_val, sample_weights_validation
        print("\nTraining set: ", len(X_train), "\nTesting set: ", len(X_test))
        protocol_dict.update({"Number of jets in the validation set": 0,})
    else:
        print("\nTraining set: ", len(X_train), "Validation set: ", len(X_val), "\nTesting set: ", len(X_test))
        protocol_dict.update({"Number of jets in the validation set": len(X_val),})
    protocol_dict.update({
        "Number of jets in the training set": len(X_train),
        "Number of jets in the testing set": len(X_test),
    })
    for arg in vars(args):
        protocol_dict.update({arg: getattr(args, arg)})

    # model selection (or reloading)
    if args.reload_nn[0]=='':
        model, subdir_name = get_model(args.model, args.number_layers, nb_features, args.activation_function, args.l1, args.l2, args.activity_l1, args.activity_l2, args.init_distr, nb_classes, args.number_maxout, args.batch_normalization)
        subdir_name +="_"+args.optimizer+"_clipn"+str(int(args.clipnorm*100))+"_"+args.objective

        part_from_inFile__reweighing_part = args.input_file.split(".h5")[0].split("__")[2]
        part_from_inFile__pTmax_part = args.input_file.split(".h5")[0].split("__")[1].split("_")[3]
        part_from_inFile__trainFrac_part = args.input_file.split(".h5")[0].split("__")[1].split("_")[2]
        part_from_inFile = part_from_inFile__trainFrac_part+"_"+part_from_inFile__pTmax_part+"__"+part_from_inFile__reweighing_part

        os.system("mkdir -p KerasFiles/%s/" % (subdir_name))
        os.system("mkdir -p KerasFiles/%s/Keras_output/" % (subdir_name))
        os.system("mkdir -p KerasFiles/%s/Keras_callback_ModelCheckpoint/" % (subdir_name))
    else:
        # TODO: check if reloading works
        # reload nn configuration from previous training
        part_from_inFile = "RE__"+args.reload_nn[0].split(".json")[0].split("__")[2]+"__"+args.reload_nn[0].split(".json")[0].split("__")[3]
        subdir_name = args.reload_nn[0].split(".json")[0].split('/')[1]
        model = model_from_json(open(args.reload_nn[0]).read())
        model.load_weights(args.reload_nn[1])

    out_str = subdir_name+part_from_inFile+"_"+backend._BACKEND+"__lr"+str(int(args.learning_rate*1000))+"_trainBS"+str(args.batch_size)+"_nE"+str(args.nb_epoch)+"_s"+str(args.seed_nr)
    if args.LRS:
        out_str.replace("__lr"+str(int(args.learning_rate*1000)),"__LRS")

    if args.validation_split > 0.:
        out_str +="_valsplit"+str(int(args.validation_split*100))+"_"
    else:
        out_str+="_val"+str(int(args.validation_set*100))

    if args.clipnorm!=0.:
        if args.optimizer=="adam":
            model_optimizer=Adam(lr=args.learning_rate, clipnorm=args.clipnorm)
        elif args.optimizer=="adamax":
            model_optimizer=Adamax(lr=args.learning_rate, clipnorm=args.clipnorm)
            model.compile(loss=args.objective, optimizer=model_optimizer, metrics=["accuracy"])
    else:
        model.compile(loss=args.objective, optimizer=args.optimizer, metrics=["accuracy"])

    # Callback: early stopping if loss does not decrease anymore after a certain number of epochs
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience)

    learning_rate_scheduler = LearningRateScheduler(lambda x: 0.1 / (1. + x))

    # Callback: will save weights (= model checkpoint) if the validation loss reaches a new minium at the latest epoch
    model_check_point = ModelCheckpoint("KerasFiles/"+subdir_name+"/Keras_callback_ModelCheckpoint/weights_"+out_str+"__{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="auto")

    training_time = time.time()
    callbacks = [model_check_point]
    if args.LRS:
        callbacks.append(learning_rate_scheduler)
    if args.patience>=0:
        callbacks.append(early_stopping)
    if backend._BACKEND=="tensorflow":
        from keras.callbacks import TensorBoard
        os.system("mkdir -p ./KerasFiles/TensorFlow_logs/")
        tensorboard = TensorBoard(log_dir='KerasFiles/TensorFlow_logs', histogram_freq=1)
        callbacks.append(tensorboard)

    if args.validation_split==0.:
        history = model.fit(X_train, Y_train,
                            batch_size=args.batch_size, nb_epoch=args.nb_epoch,
                            callbacks=callbacks,
                            show_accuracy=True, verbose=1,
                            validation_data=(X_val, Y_val, sample_weights_validation),
                            sample_weight=sample_weights_training) # shuffle=True (default)
    else:
        history = model.fit(X_train, Y_train,
                            batch_size=args.batch_size, nb_epoch=args.nb_epoch,
                            callbacks=callbacks,
                            show_accuracy=True, verbose=1,
                            validation_split=args.validation_split,
                            sample_weight=sample_weights_training) # shuffle=True (default)
    # store history:
    history_filename = "KerasFiles/"+subdir_name+"/Keras_output/hist__"+out_str+".h5"
    save_history(history, history_filename)
    training_time=time.time()-training_time

    evaluation_time = time.time()
    score = model.evaluate(X_test, Y_test,
                           show_accuracy=True, verbose=1)
    evaluation_time=time.time()-evaluation_time
    protocol_dict.update({"Classification score": score[0],"Classification accuracy": score[1]})

    prediction_time = time.time()
    predictions = model.predict(X_test, batch_size=1, verbose=1) # returns predictions as numpy array
    prediction_time=time.time()-prediction_time

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

    store = pd.HDFStore(store_str)
    store.put("predictions", pd.DataFrame(predictions))
    store.put("labels", pd.DataFrame(Y_test))
    store.close()

    print("Outputs:\n  --> saved history as:", history_filename, "\n  --> saved architecture as: KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_architecture__"+out_str+".json\n  --> saved NN weights as: KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_weights__"+out_str+".h5\n  --> saved predictions as ", store_str)


def _get_args():
    help_input_file = "Input file determining the pT and eta ranges as well as the c-fraction in the BG sample (default: %(default)s)."
    help_reload_nn = "Reload previously trained model, provide architecture (1st argument; JSON) and weights (2nd argument; HDF5) (default: %(default)s)."
    help_batch_size = "Batch size: Set the number of jets to look before updating the weights of the NN (default: %(default)s)."
    help_nb_epoch = "Set the number of epochs to train over (default: %(default)s)."
    help_init_distr = "Weight initialization distribution determines how the initial weights are set (default: %(default)s)."
    help_seed_nr = "Seed initialization (default: %(default)s)."
    help_validation_set = "Part of the training set to be used as validation set (default: %(default)s)."
    help_model = "Architecture of the NN. Due to the current code structure only a few are available for parameter search (default: %(default)s)."
    help_activation_function = "activation function (default: %(default)s)."
    help_learning_rate = "Learning rate used by the optimizer (default: %(default)s)."
    help_optimizer = "optimizer for training (default: %(default)s)."
    help_objective = "objective (or loss) function for training (default: %(default)s)."
    help_l1 = "L1 weight regularization penalty (default: %(default)s)."
    help_l2 = "L2 weight regularization penalty (default: %(default)s)."
    help_activity_l1 = "L1 activity regularization (default: %(default)s)."
    help_activity_l2 = "L2 activity regularization (default: %(default)s)."

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-in", "--input_file", type=str,
                        default="PreparedSample__V47full_Akt4EMTo_bcujets_pTmax300GeV_TrainFrac85__b_reweighting.h5",
                        help=help_input_file)
    parser.add_argument("-r", "--reload_nn",
                        type=str, nargs='+', default=["",""],
                        help=help_reload_nn)

    parser.add_argument("-m", "--model",
                        type=str, default="Dense",
                        choices=["Dense", "Maxout_Dense"],
                        help=help_model)

    parser.add_argument("-obj", "--objective",
                        type=str, default="categorical_crossentropy",
                        choices=["categorical_crossentropy", "mse"],
                        help=help_objective)
    parser.add_argument("-o", "--optimizer",
                        type=str, default="adam",
                        choices=["adam", "adamax"],
                        help=help_optimizer)

    parser.add_argument("-lr", "--learning_rate",
                        type=float, default=0.001,
                        help=help_learning_rate)

    parser.add_argument("-l1", "--l1",
                        type=float, default=0.,
                        help=help_l1)
    parser.add_argument("-l2", "--l2",
                        type=float, default=0.,
                        help=help_l2)
    parser.add_argument("-al1", "--activity_l1",
                        type=float, default=0.,
                        help=help_activity_l1)
    parser.add_argument("-al2", "--activity_l2",
                        type=float, default=0.,
                        help=help_activity_l2)

    parser.add_argument("-af", "--activation_function",
                        type=str, default="relu",
                        choices=["relu", "tanh", "ELU", "PReLU", "SReLU"],
                        help=help_activation_function)
    parser.add_argument("-bs", "--batch_size",
                        type=int, default=80,
                        help=help_batch_size)
    parser.add_argument("-nl", "--number_layers",
                        type=int, default=5,
                        choices=[5,4,3],
                        help="number of hidden layers (default: %(default)s).")
    parser.add_argument("-nm", "--number_maxout",
                        type=int, default=0,
                        help="number of Maxout layers running in parallel")
    parser.add_argument("-p", "--patience",
                        type=int, default=5,
                        help="number of epochs witout improvement of loss before stopping")
    parser.add_argument("-ne", "--nb_epoch",
                        type=int, default=20,
                        help=help_nb_epoch)
    parser.add_argument("-id", "--init_distr",
                        type=str, default="glorot_uniform",
                        help=help_init_distr)
    parser.add_argument("-sn", "--seed_nr",
                        type=int, default=12264,
                        help=help_seed_nr)
    parser.add_argument("-vs", "--validation_set",
                        type=float, default=.1,
                        help=help_validation_set)
    parser.add_argument("-cn", "--clipnorm",
                        type=float, default=0.,
                        help="clipnorm for gradient clipping (default: %(default)s).")
    parser.add_argument("-vsp", "--validation_split",
                        type=float, default=0.,
                        help="validation split using part of the training set (default: %(default)s)")
    parser.add_argument("-bn", "--batch_normalization", action="store_true",
                       help="normalize between layers for each batch")
    parser.add_argument("-lrs", "--LRS", action="store_true",
                       help="use Learning Rate Scheduler")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    _run()
