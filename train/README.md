# Guide to `manager.py`

This is a (hopefully) simple guide to the manager.py script and its purpose

## Purpose

The project is structured to be used in the following way:

1. Create a directory for a specific setup, like (`train/IAD_setup`)
2. In that directory, create `main.py`, `config.py` and `plot.py`
   1. `main.py` contains the entry point and the code for creating a model, setting up its callbacks, training data and other things required for the `Model.fit` call, execute the model training and testing. It should do this based on the parameters in `config.py` and then plot the results using `plot.py`. 
   2. `plot.py` contains the code for plotting the results of the training
   3. `config.py` contains the parameters for the `main.py` script (such as model type, data preprocessing parameters, architecture details and other training hyper parameters.) 

3. Use `manager.py create IAD_setup` to create a run directory for the specified setup. The manager will parse the configuration script/file (see below) and copy it, along with the main and plot scripts to a directory with a unique name, specified by the `TAG` variable in the config file, the `name` attribute of the grid search (if there is any) and a run-id (which is tracked automatically by the manager). Specifically, the path is as follows `RUN_DIR=[OUTPUT_DIR]/R[RUN_ID]_[TAG]` where `OUTPUT_DIR` and `TAG` are set in the configuration script (see below) and `RUN_ID` is obtained from the `RUN_ID_FILE` (see below) or from the command-line argument `--run-id`. If you use repeated runs (for ensembles), there will be a folder for each repeat (named 0,1,...,#repeats), each containing a config file for that specific repeat (usually this means a different `SEED`)
4. Run the automatically created jobscripts in `RUN_DIR`. If you specified a `GRID` each configuration in the grid will have its own folder and jobscript. These jobscripts will be listed in `jobscripts.lst` in `RUN_DIR`. 

Note: the manager renames all `.py` scripts that it copies to `.m.py` files. This does not change anything about the files themselves, but you can setup your IDE (like VSCode) to use a different icon for `.m.py` files, to remind you that these are copied files meant for archive purposes and shouldn't be modified.

## Commandline arguments

Usage `python3 manager.py create [-h] [--script SCRIPT] [--config CONFIG] [--plot PLOT] [--no-grid] [--run-id RUN_ID] [--extra-files [EXTRA_FILES ...]] [--plot-ens [PLOT_ENS]] model-dir`

with 

- `model-dir` : setup directory ('IAD_setup' in the example above)
- `--script [script_name]`: the name of the main script (usually `main.py`)
- `--config [config_name]`: the name of the config script (usually `config.py`)
- `--plot [plot_name]`: the name of the plot script (usually `plot.py`)
- `--run-id [id]`: sets the RUN_ID manually, instead of using `run_id.json` specified by `RUN_ID_FILE` in the config script (see below).
- `--extra-files`: lists extra files that you want to be copied to `RUN_DIR`
- `--plot-ens`: specifies a script to use for plotting ensembles. If specified, the manager will automatically add a bit to the jobscripts to run the ensemble plotting script when the last repeat is finished.

## Configuration files

In this project, configuration files are used to set the hyperparameters of `main.py`. This configuration file can be either a python script or a json file. Whereas a JSON file is just treated as a dictionary, the a python file has extra benefits, as described below. The manager will parse a configuration script and turn it into a JSON configuration file that will be copied to the run directory of the current run.

The configuration file should specify the following variables:

- `JOBNAME`: determines the name of the parent directory of this run (so you can have lots of runs with the same `JOBNAME` , such that `JOBNAME` kind of functions as a project name)
- `RUN_ID_FILE`: the file used to track the run-id. The manager will automatically increase the run-id by 1 every time a new run is created.
- `OUTPUT_DIR`: specifies the root directory of all outputs
- `TAG`: specifies the name for this specific run.
- `REPEATS`: sets the number of repeats (or, ensemble size) of a run. If you don't want a repeated run, set it to `None`. If `REPEATS=3` for instance, the manager will produce three configurations that differ only in their value of `SEED`
- `SLURM`: a dictionary that contains the following keys:
  - `ACCOUNT`: the slurm account to be used for the job
  - `PARTITION`: the partition to be used for the job
  - `MEMORY`: the value of `--mem-per-cpu`
  - `TIME`: the value of `--time`
  - (optionally): `GPUS`: the number of GPUs to use
  - (optionally): `LOGDIR`: the directory to save the stdout and stderr logs in. Use 'auto' to set this to the `run_dir/logs`.

### Grid search

If you want to perform a grid search, or test out various values for a parameters, you can add a parameter named `GRID` that is a list of dictionaries, each of which contains a 'name' and a set of parameter names with values to override the rest of the config file. Most config scripts do this using the `create_grid` function. For example,

```python
GRID = create_grid(lambda DATA_SEED: f"{DATA_SEED=:d}", {"DATA_SEED": [0,1,2,3,4]})
```

will cause the manager to create a grid of 5 configurations, each with a different data seed. Note that it is important to consider which variables need to be changed as a result of a different value for the grid parameters. (Changing the model name might, for example, require a different model-specific configuration extension)

### Conditional values

Say you have several datasets, `data-N50.h5`, `data-N100.h5`, and `data-N200.h5`  that had a different pre-processing depending on some parameter `N` (in this case, that is the number of jet constituents per jet.). The model trained in `main.py` also requires this hyperparameter, so naturally, it is part of your config file. How can you make sure that the config file will always refer to the correct data, without having to manually check that if `N=50`, `data_file="data-N50.h5"`, for example. For this, you can use conditional variables!

If you assign a string of the form "$<|...|>" to a variable in a configuration script, the manager will treat the stuff between `<|` and `|>` as python code, run it using `eval` and replace that string with the result. So, in this case, you could use

```python
N=50
#other parameters
data_file="data-N$<|N|>.h5"
```

Of course, you could also use python format-strings (like `f"data-N{N}.h5"`) in this case, but the advantage of using conditional variables is that they are assigned _after_ the rest of the configuration file has been parsed, including the GRID parameter. So, if you have a grid that contains several values of `N`, each grid configuration will automatically have the correct value for `data_file`!

### Extra configurations

If you have several models, each of which has a whole bunch of model-specific hyperparameters (like layer sizes, training schedule, etc.), you could pack them all in a single config file and name them such that only those for the chosen model get used by `main.py` whereas the rest simply gets ignored. You could even use a `match-case` or `if-elif-else` structure to set parameters based on the value of the `MODEL` variable.

However, an easier and more readable option is to use so-called 'extra configurations'. This simple involves putting all model-specific variables in their own config file, say `model_simple.py` and `model_complex.py` and adding the name of which config file to 'append' to the configuration in the `EXTRA_CONFIGS` variable in the main configuration script. The manager will automatically retrieve all files listed in `EXTRA_CONFIGS`, parse them as well and add them to the parsed configuration. You can even use parameters declared in the main configuration script in conditional variables in extra configurations.

For example,

```python
MODEL="simple"

EXTRA_CONFIGS = []
if(MODEL=="simple"):
    EXTRA_CONFIGS.append("config_simple.py")
elif(MODEL=="complex"):
    EXTRA_CONFIGS.append("config_complex.py")
```





