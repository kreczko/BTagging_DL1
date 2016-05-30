'''
Script to construct different architectures and configurations, i.e. Keras models
'''
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.regularizers import l1l2, l1, activity_l1, activity_l1l2
from keras.layers.core import MaxoutDense, Dense, Activation, Dropout, Highway
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU, SReLU

def get_model(model, nb_layers, nb_features, activation_function, l1, l2, activity_l1, activity_l2, init_distr, nb_classes, number_maxout, batch_normalization):
    '''
    This function defines different models and returns the model and name of the subdirectory
    '''
    # "Dense" architecture using only fully connected layers
    def _get_model_Dense(nb_features, nb_layers, activation_function, l1, l2, activity_l1, activity_l2, init, nb_classes, batch_normalization):

        name = "D_"+str(nb_classes*2**4)
        if nb_layers==5 or nb_layers==4:
            if nb_layers==5:
                name+="_"+str(nb_classes*2**4)
            name+="_"+str(nb_classes*2**4)+"_"+str(nb_classes*2**3)+"_"+str(nb_classes*2**2)+"_"+str(nb_classes*2)
        elif nb_layers==3:
            name+="_"+str(nb_classes*2**3)+"_"+str(int((nb_classes*2**2+nb_classes*2)*0.5))
        name+="_"+str(nb_classes)
        if batch_normalization:
            name+="__BN_"
        name+=activation_function

        model = Sequential()
        if activation_function in ["relu", "tanh"]:
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(Dense(nb_classes*2**4, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(Dense(nb_classes*2**4, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2+nb_classes) * 0.5), activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                model.add(Dense(nb_classes*2**4, activation=activation_function, init=init, input_shape=(nb_features,)))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(Dense(nb_classes*2**4, activation=activation_function, init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, activation=activation_function, init=init))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, activation=activation_function, init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, activation=activation_function, init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), activation=activation_function, init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        elif activation_function=="SReLU":
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(Dense(nb_classes*2**4, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(Dense(nb_classes*2**4, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                model.add(Dense(nb_classes*2**4, init=init, input_shape=(nb_features,)))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(Dense(nb_classes*2**4, init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        elif activation_function=="ELU":
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(Dense(nb_classes*2**4, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_classes==5:
                    model.add(Dense(nb_classes*2**4, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                model.add(Dense(nb_classes*2**4, init=init, input_shape=(nb_features,)))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_classes==5:
                    model.add(Dense(nb_classes*2**4, init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        elif activation_function=="PReLU":
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(Dense(nb_classes*2**4, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_classes==5:
                    model.add(Dense(nb_classes*2**4, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                model.add(Dense(nb_classes*2**4, init=init, input_shape=(nb_features,)))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_classes==5:
                    model.add(Dense(nb_classes*2**4, init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(nb_classes*2, init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        return model, name


    # "Maxout_Dense" architecture using only fully connected layers
    def _get_model_Maxout_Dense(nb_features, nb_layers, number_maxout, activation_function, l1, l2, activity_l1, activity_l2, init, nb_classes, batch_normalization):
        name = "MO"+str(number_maxout)+"_"+str(nb_classes*2**4)
        if nb_layers==5 or nb_layers==4:
            if nb_layers==5:
                name+="_"+str(nb_classes*2**4)
            name+="_"+str(nb_classes*2**4)+"_D_"+str(nb_classes*2**3)+"_"+str(nb_classes*2**2)+"_"+str(nb_classes*2)
        elif nb_layers==3:
            name+="_D_"+str(nb_classes*2**3)+"_"+str(int((nb_classes*2**2+nb_classes*2)*0.5))
        name+="_"+str(nb_classes)
        if batch_normalization:
            name+="__BN_"
        name+=activation_function

        model = Sequential()
        if activation_function in ["relu", "tanh"]:
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout,  init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout,  init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), activation=activation_function, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                print number_maxout, nb_classes, init, nb_features
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout,  init=init, input_shape=(nb_features,)))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout,  init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, activation=activation_function, init=init))
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, activation=activation_function, init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, activation=activation_function, init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), activation=activation_function, init=init))
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        elif activation_function=="SReLU":
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, input_shape=(nb_features,)))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())

                model.add(Dense(nb_classes*2**3, init=init))
                model.add(SReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init))
                    model.add(SReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        elif activation_function=="ELU":
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, input_shape=(nb_features,)))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init))
                model.add(ELU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init))
                    model.add(ELU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        elif activation_function=="PReLU":
            if l1!=0. or l2!=0. or activity_l1!=0. or activity_l2!=0.:
                name+="_l1"+str(int(l1*100))+"_l2"+str(int(l2*100))+"_al1"+str(int(activity_l1*100))+"_al2"+str(int(activity_l2*100))
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2), input_shape=(nb_features,)))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(nb_classes*2, init=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2)))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))
            else:
                model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init, input_shape=(nb_features,)))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5:
                    model.add(MaxoutDense(output_dim=nb_classes*2**4, nb_feature=number_maxout, init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes*2**3, init=init))
                model.add(PReLU())
                if batch_normalization:
                    model.add(BatchNormalization())
                if nb_layers==5 or nb_layers==4:
                    model.add(Dense(nb_classes*2**2, init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                    model.add(Dense(nb_classes*2, init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                elif nb_layers==3:
                    model.add(Dense(int((nb_classes*2 + nb_classes)*0.5), init=init))
                    model.add(PReLU())
                    if batch_normalization:
                        model.add(BatchNormalization())
                model.add(Dense(nb_classes, activation="softmax", init=init))

        return model, name




    if(model=="Dense"):
        model, subdir_name = _get_model_Dense(nb_features, nb_layers, activation_function, l1, l2, activity_l1, activity_l2, init_distr, nb_classes, batch_normalization)
    elif(model=="Maxout_Dense"):
        model, subdir_name = _get_model_Maxout_Dense(nb_features, nb_layers, number_maxout, activation_function, l1, l2, activity_l1, activity_l2, init_distr, nb_classes, batch_normalization)
    return model, subdir_name
