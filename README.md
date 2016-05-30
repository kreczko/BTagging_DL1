What is this?
-------------

This is the code required for

1. preprocessing to prepare the MC b-tagging samples for training and testing (via `scripts/PrepareSamples.py`)

1. the training and testing itself (via `scripts/btagging_nn.py`, with the explicit architecture constructed in `scripts/btagging_nn_models.py`)

1. few basic plotting scripts to check the training.

(Tested using Python 2.7.10)


How do I use it?
----------------

#### Quick Start ####

###### Setup

The packages are available within Anaconda but Anaconda builds against libstdc++, the GCC standard library. On Mac OS X 7 and later the default compiler links against libc++, the Clang C++ standard library. therefore i switched Anaconda off and installed everything separately

Package requirements:

* theano (https://github.com/Theano/Theano) or TensorFlow (https://www.tensorflow.org)

* Keras (https://github.com/fchollet/keras)

* pandas (http://pandas.pydata.org)

* matplotlib (http://matplotlib.org)

* numpy

* HDF5 (http://www.hdfgroup.org/HDF5)

* root_numpy (https://github.com/ndawe/root_numpy)




The initial ROOT ntuple has to be stored in a subdirectory called `inputFiles/`.


###### Usage

For the **initial preprocessing of the data**, please execute:

`./scripts/PrepareSamples.py '<path-to-input_ROOT-file>'`

In case the HDF5-file has already been created for the ROOT-file in an earlier run and one just wants to e.g. use a different reference distribution for (eta, pT) reweighting, use '' as argument for '<input_ROOT-file>' to directly load the data frame from the HDF5-file and save time. The default filename which is used then is defined by `default_sample_info` in `scripts/btag_nn_inputs.py`.


_Example_:

`./scripts/PrepareSamples.py "inputFiles/<flat-ntuple-file>.root"`


Then, to **train**, execute:

`./scripts/btagging_nn.py -in '<path-to-prepared_HDF5-file>'`

_Example_:

`./scripts/btagging_nn.py -m "Dense" -nl 3 -bs 500 -ne 1 -p -1 -vs 0.3 -in "PreparedFiles/PreparedSample__V47full_Akt4EMTo_bcujets_pTmax300GeV_TrainFrac85__b_reweighting.h5"

`./scripts/btagging_nn.py -m "Maxout_Dense" -nl 3 -nm 20 -bs 500 -ne 1 -p -1 -vs 0.3 -in "PreparedFiles/PreparedSample__V47full_Akt4EMTo_bcujets_pTmax300GeV_TrainFrac85__b_reweighting.h5"

####### Few remarks

* `-p -1` will not use EarlyStopping

* when using the model "Maxout_Dense", it is necessary to indicate the number of layers that are to be trained in parallel using `-nm` e.g. `-nm 10`


More information on the implemented flags is available via the default argparse `--help` option, i.e. `./scripts/PrepareSamples.py --help` and `./scripts/btagging_nn.py --help`.
