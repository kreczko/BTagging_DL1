What is this?
-------------

This is the code required for

1. preprocessing to prepare the MC b-tagging samples for training and testing

1. the training and testing itself.


(Tested using Python 2.7.10)


How do I use it?
----------------

#### Quick Start ####

Make `btag_nn_inputs` and `btag_nn` module functions globally available in python:

From `scripts/btag_nn_inputs/`: `sudo python setup.py install`

From `scripts/btag_nn/`: `sudo python setup.py install`

`python scripts/PrepareSamples.py '<path-to-input_ROOT-file>'`

In case the HDF5-file has already been created for the ROOT-file, use '' as argument for '<input_ROOT-file>' to directly load the data frame from the HDF5-file and save time.

The initial ROOT ntuple has to be stored in a subdirectory called _inputFiles/_

_Example:_

`python scripts/PrepareSamples.py "inputFiles/mc15_13TeV_V33full_Akt4EMTo_minibtag.root"`


Then, to train, do

`python scripts/btagging_nn.py -i '<path-to-prepared_HDF5-file>'`

_Example:_

`python scripts/btagging_nn.py "PreparedFiles/PreparedSample__jet_ntuple_TrainFrac80_cFrac20.h5"`


More Help available via `python scripts/PrepareSamples.py --help` and `python scripts/btagging_nn.py --help`.



#### Requirements for this to work: ####

The packages are available within Anaconda but Anaconda builds against libstdc++, the GCC standard library. On Mac OS X 7 and later the default compiler links against libc++, the Clang C++ standard library. therefore i switched Anaconda off and installed everything separately

Via pip install:

* theano (https://github.com/Theano/Theano)

* Keras (https://github.com/fchollet/keras)

* pandas (http://pandas.pydata.org)

* matplotlib (http://matplotlib.org)

Via homebrew (with python3 support)

* numpy

* HDF5 (http://www.hdfgroup.org/HDF5)

Via git clone and then make install:

* root_numpy (https://github.com/ndawe/root_numpy)


Support for converstion from ROOT to pandas:

`git clone https://github.com/mickypaganini/YaleATLAS.git`

`python setup.py`

`build sudo python setup.py install`
