'''

    Script to construct/store different architectures and configurations, i.e. Keras models

'''

def get_model(model, nb_features, activation_function, init_distr, nb_classes):    
    '''
    This function defines different models and returns the model and name of the subdirectory
    '''
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.utils import np_utils

    def _get_model_Dense(nb_features, activation_function, init, nb_classes):
        name = "D_48_24_9_3__"+activation_function
        model = Sequential()
        model.add(Dense(48, activation=activation_function, init=init, input_shape=(nb_features,)))
        model.add(Dense(24, activation=activation_function, init=init))
        model.add(Dense(9, activation=activation_function, init=init))
        model.add(Dense(nb_classes, activation="sigmoid", init=init))
        return model, name

    if(model=="Dense"):
        model, subdir_name = _get_model_Dense(nb_features, activation_function, init_distr, nb_classes)
    return model, subdir_name
