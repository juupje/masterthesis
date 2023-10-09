# Thesis project

This repository contains the source code used for my master thesis on the topic of weakly supervised anomaly detection at the LHC.

The code has been written with a somewhat general use in mind, though some parts are written specifically for my needs.

## Usage
The folder `lib` contains various scripts and modules that are used throughout the code.

- The Deep Learning models, including TensorFlow implementations of ParticleNet [1,2], LorentzNet [3,4] and PELICAN [5,6], can be found in `lib/models`
- `lib/utils` contains a module containing a bunch of utility scripts and functions

The rest of the code assumes that `lib` is in Python's `PATH`.

The folder `processing` contains several scripts for processing the raw data files, such as merging LHCO and specific numpy files and  converting ROOT files to LHCO format.

Finally, the scripts used for model training are found in `train`. Each type of setup used in my thesis has a separate directory, though most code is shared between them. Each setup consists of four main scripts:

- `main.py` contains the main script (surprise!) which takes care of loading the data, constructing the Keras model, running the `model.fit()` function, storing the training curves as well as applying the model to test data and storing those results.
- `plot.py` plots the results of a model trained by `main.py` . It plots the training curves, learning rate schedule, and the score distribution and ROC curve of the test data. 
- `plot-ens.py` plots an ensemble of models, each trained individually by `main.py` with different output directories (and different configs).
- `config.py` contains the parameters used by `main.py` and `plot.py` to train, test, and plot the models. It contains the (part of) the training hyperparameters, logging parameters and the paths of the datasets. Depending on which architecture is used, it refers to additional config files containing architecture-specific hyperparameters (and the learning rate schedule).

To help with maintaining a useful structure of model runs, the script `train/manager.py` can take a setup directory, parse its config file and copy the scripts to an 'output' directory. It additionally features the possibility of create many individual runs with different parameters (such as a grid search). It can also create multiple identical models (specified by `REPEATS` in the config script) to create ensemble runs.

The manager script will also create slurm batch scripts for each run (repeats use array jobs).

## Notes

The code uses the `steganologger` [script](https://github.com/juupje/steganologger) and [VSCode extension]([https://marketplace.visualstudio.com/items?itemName=Juupje.steganologger) for logging config files in plots (which is very useful when you'd like to check the config file used to create a plot).

# References

[1] H. Qu and L. Gouskos. “Jet tagging via particle clouds”. In: Physical Review D 101.5 (Mar. 2020). doi: 10.1103/physrevd.101.056019. 

[2] H. Qu and L. Gouskos. ParticleNet. GitHub. Commit 597f0c5. Oct. 2019. https:// github.com/hqucms/ParticleNet/. 

[3] S. Gong et al. “An efficient Lorentz equivariant graph neural network for jet tagging”. In: Journal of High Energy Physics 2022.7 (July 2022). doi: 10.1007/jhep07(2022)030. 

[4] S. Gong et al. LorentzNet-release. Github. Commit 14bbe78. July 2022. https://github. com/sdogsq/LorentzNet-release. 

[5] A. Bogatskiy et al. PELICAN: Permutation Equivariant and Lorentz Invariant or Covariant Aggregator Network for Particle Physics. Nov. 2022. doi: 10.48550/arXiv.2211. 00454. arXiv: 2211.00454 [hep-ph]. 

[6] A. Bogatskiy et al. PELICAN. GitHub. Commit 3b7a93a. Nov. 2022. https://github. com/abogatskiy/PELICAN.

