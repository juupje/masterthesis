#!/usr/bin/env python3
import os
import argparse
from utils import configs, misc
import re
from datetime import datetime
import subprocess
import json

SBATCH_TEMPLATE = \
"""#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --output={logdir}/{logindexing}.out  # Save the log files with job name, array job id, and task id
#SBATCH --error={logdir}/{logindexing}.err   # Save error files with job name, array job id, and task id
#SBATCH --mem-per-cpu={memory}      # Memory per cpu
#SBATCH --gres=gpu:{gpus}                # Number of GPUs per node
#SBATCH --time={time}               # Max wall time
#SBATCH --mail-type=END,FAIL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=joep.geuskens@rwth-aachen.de # Email address for notifications
{sbatch_additions}

{script_additions_pre}

WORKDIR="{workdir}"
RUNDIR="{rundir}"
SCRIPTNAME={scriptname}
CONFIG="{configname}"
PLOT_SCRIPT={plotscript}

source $HOME/.zshrc
cd $WORKDIR
conda activate {condaenv}

{script_additions_post}
python3 "$SCRIPTNAME" --outdir="$RUNDIR" --config="$CONFIG" {extra_args}"""

PLOT_ENS_TEMPLATE = \
"""
EXIT_CODE=$?

# If the Python script failed, exit with the same exit code
if [ $EXIT_CODE -ne 0 ]; then
    echo "Python script failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

if [ $SLURM_ARRAY_TASK_ID -eq {njobs} ]
then
    sbatch --dependency=afterok:$SLURM_JOB_ID --time 5 --mem-per-cpu 2G --output={logdir}/$SLURM_JOB_ID-plot.out \
        --wrap "source $HOME/.zshrc; conda activate {condaenv}; python3 {plot_ens_file} --outdir {outdir} -l final --save"
fi
"""

EXT = "m.py"
BASH_GET_SCRIPT_DIR = '"$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";'
def confirm(question="Accept guesses?"):
    answer = "x"
    while answer not in "yn":
        answer = input(f"{question} (y/n) ").lower()
    return answer == "y"
        
def guessscript(file_list, name, log_name,require=True):
    files = list(filter(lambda s: name in s, file_list))
    if(len(files)==1):
        file = files[0]
        files.remove(file)
        print(f"Guessed {log_name} script: {file}")
        return os.path.basename(file)
    else:
        print(f"Cannot guess {log_name} script, possible candidates are: ")
        print("\t", files)
        if(require):
            exit()

def guess_files(args, which=['config', 'plot', 'script']):
    print("Guessing " + ", ".join(which))
    #Guessing time!
    import glob
    if('jobscript' in which):
        shFiles = glob.glob('*.sh')
        if(len(shFiles)==1):
            args["jobscript"] = os.path.basename(shFiles[0])
            print("Guessed jobscript: " + shFiles[0])
        elif("jobscript.sh" in shFiles):
            args["jobscript"] = os.path.basename("jobscript.sh")
            print("Guessed jobscript: jobscript.sh")
        else:
            print("Cannot guess jobscript, possible candidates:")
            print("\t",shFiles)
            exit()
        which.remove('jobscript')

    pyFiles = glob.glob("*.py")
    for s in which:
        if(len(pyFiles)>=2):
            if(s == 'plot'):
                args[s] = guessscript(pyFiles, s, "plotting", require=False)
            elif(s == 'config'):
                args[s] = guessscript(pyFiles, s, s)
            elif(s == 'script'):
                args[s] = guessscript(pyFiles, 'main', 'main')
        else:
            print("Cannot guess " + s)
            exit()
    if not confirm(): exit()

def guess_config(args):
    import glob
    configFiles = glob.glob("*config.*")
    args["config"] = guessscript(configFiles,"config", "config")
    if not confirm(): exit()

def write_version_info(outdir):
    with open(os.path.join(outdir, "version.txt"), 'w') as f:
        f.write("Created on: " + datetime.now().strftime(r"%Y-%m-%d %H:%M:%S") + "\n")
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        f.write("With Git commit: " + git_hash + "\n")

def create_jobscript_and_config(outdir:str, item_dir:str, script_name:str, configuration:dict, counter:int,
                                repeats:int=None, kfolds:int|None=None) -> str:
    global SBATCH_TEMPLATE
    #create the output for this grid item
    fulldir = os.path.join(outdir, item_dir)
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    if(repeats):
        for repeat in range(repeats):
            configuration["SEED"] += 1 #increase seed (we never want a repeat with the same seed!)
            #write config to a file
            os.makedirs(os.path.join(fulldir, str(repeat)),exist_ok=True)
            with open(os.path.join(fulldir, str(repeat), f"config.json"), "w") as f:
                json.dump(configuration, f, ensure_ascii=True, indent=2)
    else:
        with open(os.path.join(fulldir, f"config.json"), "w") as f:
            json.dump(configuration, f, ensure_ascii=True, indent=2)
    #create jobscript
    slurm_config = configuration["SLURM"]
    if slurm_config.get("LOGDIR", "auto") == "auto":
        slurm_config["LOGDIR"] = os.path.join(fulldir, "logs")
    if(not os.path.isdir(slurm_config["LOGDIR"])):
        os.makedirs(slurm_config["LOGDIR"])
    

    format_dict = dict(jobname=f"{configuration['JOBNAME']:s}_{configuration['TAG']:s}_R{RUN_ID:d}"+\
                                                (f"-{counter:d}" if counter is not None else ""),
                        account=slurm_config.get("ACCOUNT",None) or "",
                        partition=slurm_config["PARTITION"],
                        logdir=slurm_config["LOGDIR"],
                        logindexing="%A_%a" if repeats or kfolds else "%J",
                        memory=slurm_config["MEMORY"],
                        time=slurm_config["TIME"],
                        workdir=outdir+("/" if len(outdir)>0 else ""),
                        rundir=f"{item_dir}" if len(item_dir)>0 else "",
                        condaenv=slurm_config["CONDA_ENV"],
                        gpus=slurm_config.get("GPUS", 1),
                        scriptname=f"{script_name:s}.{EXT:s}",
                        configname="$WORKDIR$RUNDIR/config.json",
                        plotscript=f"{os.path.splitext(args['plot'])[0]:s}.{EXT}" if args["plot"] else "",
                        extra_args='--plot_script="$PLOT_SCRIPT"' if args["plot"] else "", 
                        sbatch_additions="",
                        script_additions_pre="",
                        script_additions_post="unset CUDA_VISIBLE_DEVICES")
    if len(format_dict["account"])==0:
        SBATCH_TEMPLATE = SBATCH_TEMPLATE.replace("#SBATCH --account=", "")
    if(repeats is not None or kfolds is not None):
        format_dict["sbatch_additions"] += f"#SBATCH --array=0-{(repeats or 1)*(kfolds or 1)-1:d}\n"
        format_dict["script_additions_pre"] += f"REPEAT=$((SLURM_ARRAY_TASK_ID%{repeats or 1}))\n" + \
                                             f"FOLD=$((SLURM_ARRAY_TASK_ID/{repeats or 1}))\n"
        if(repeats):
            format_dict["rundir"] = os.path.join(str(item_dir), "${REPEAT}") #directory with the config file
        if(kfolds):
            format_dict["extra_args"] += " --fold=${FOLD}"

    #write jobscript to file
    jobscript_name = os.path.join(fulldir, "jobscript.sh")
    with open(jobscript_name, 'w') as f:
        f.write(SBATCH_TEMPLATE.format(**format_dict))
        if args["plot_ens"]:
            plot_ens_dict = dict(njobs=(repeats or 1)*(kfolds or 1)-1,
                                 logdir=slurm_config["LOGDIR"],
                                 outdir=format_dict["workdir"]+str(item_dir),
                                 condaenv=slurm_config["CONDA_ENV"],
                                 plot_ens_file=os.path.splitext(args['plot_ens'])[0]+".m.py")
            f.write("\n"+PLOT_ENS_TEMPLATE.format(**plot_ens_dict))
    return jobscript_name, (repeats or 1)*(kfolds or 1)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Batch job manager")

    sp = parser.add_subparsers(help="commands", dest="command")

    p1 = sp.add_parser("create")

    p1.add_argument("model-dir", help="Model directory", type=str)
    p1.add_argument("--script","-s", help="script name (in model directory)", type=str, required=False)
    p1.add_argument("--config", "-c", help="configuration script name (in model directory)", type=str, required=False)
    p1.add_argument("--plot", "-p", help="plotting script name (in model directory)", type=str, required=False)
    p1.add_argument("--no-grid", "-n", help="Create a single job, no grid", action='store_true')
    p1.add_argument("--run-id", "-r", help="Use this specific run id", type=int)
    p1.add_argument("--extra-files", "-f", help="Extra files to be copied", type=str, nargs='*', default=[])
    p1.add_argument("--plot-ens", help="Include ensemble plotting script", nargs='?', const='plot-ens.py')
    p1.add_argument("--old-format", "-o", help="Use legacy format for backwards compatibility", action='store_true')

    p2 = sp.add_parser("run")
    p2.add_argument("dir", help="directory to run (relative to the outputs folder)", type=str)
    p2.add_argument("--output-dir", help="output directory, of which `dir` is a subdirectory (default: $HOME/thesis/outputs)", type=str, default=os.path.join(os.getenv("HOME"), "thesis", "outputs"))
    p2.add_argument("--no-sbatch", "-s", help="Run the jobscript with `sh`. Only applicable to single jobs", action='store_true')

    args = vars(parser.parse_args())

    if(args["command"]=="create"):
        os.chdir(args["model-dir"])
        if args["plot_ens"] and args["plot_ens"] not in args["extra_files"]:
            args["extra_files"].append(args["plot_ens"])
        #see if we need to guess the scripts
        scripts = ['config', 'plot', 'script']
        copy = scripts.copy()
        for s in copy:
            if(args[s]):
                scripts.remove(s)
        if(len(scripts)>0):
            guess_files(args, which=scripts)
        
        config = configs.parse_config(os.path.join(args["config"]), expand=False)
        print(config)
        change_seed = config["CHANGE_SEED"]
        
        #confirm if the user agrees with the grid
        use_grid = ("GRID" in config and not args["no_grid"])
        if(use_grid):
            names = config["GRID"][0].keys()
            lens = {name:max([len(str(item[name])) for item in config["GRID"]]) for name in names}
            lens = [max(2,len(n), lens[n]) for n in names]
            print(" # "+" ".join([f"{name:{x:d}s}" for name,x in zip(names,lens)]))
            for i,item in enumerate(config["GRID"]):
                print(f"{i:2d} "+" ".join([f"{str(item[name]):{x:d}s}" for name,x in zip(names,lens)]))
            if not confirm("Accept grid?"): exit()
        
        #check if we're dealing with k-fold cross validation
        use_kfcv = "CROSS_VALIDATION" in config
        if(use_kfcv):
            print("Using k-fold cross validation!")

        #log the ID of the current run
        if(args["run_id"]):
            RUN_ID = args["run_id"]
        else:
            RUN_ID = misc.get_run_id(config["RUN_ID_FILE"])
            print(f"Using run ID {RUN_ID}")
        
        if args["old_format"]:
            outdir = os.path.join(config["OUTPUT_DIR"], config["TAG"]+f"_R{RUN_ID:d}")
            if(use_grid): outdir += "g"
        else:
            outdir = os.path.join(config["OUTPUT_DIR"], f"R{RUN_ID:d}"+("g" if use_grid else "")+"_"+config["TAG"])

        #Create the output dir
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        elif not confirm("Output directory already exists.\nDo you want to overwrite?"):
            exit()
        
        write_version_info(outdir)

        script_basename = os.path.basename(args["script"]).replace(".py", "")
        config["RUN_ID"] = RUN_ID
        #copy all required scripts into the output folder
        for file in [args['script'], args['plot']] + ([args['config']] if use_grid else []) + args["extra_files"]:
            if(file is None): continue
            assert(os.path.isfile(file)), f"File '{file}' does not exist"
            split = os.path.splitext(file)
            if(split[1]=='.py'):
                os.system(f'cp "{file}" "{outdir:s}/{split[0]}.{EXT:s}"')
            else:
                os.system(f'cp "{file}" "{outdir:s}/{file}"')
        
        if(use_grid):
            from copy import deepcopy
            scripts_to_run = []
            counter = 0
            jobcounter = 0
            grid = deepcopy(config["GRID"])
            del config["GRID"]
            for item in grid:
                base_config = deepcopy(config)
                #copy the grid item into the base config
                for k, v in item.items():
                    base_config[k.upper()] = v
                if(config["CHANGE_SEED"]):
                    base_config["SEED"] += counter #change the seed for this item in the grid
                #expand config
                configs.expand_config_dict(base_config, args["config"])
                
                '''
                #copy original config file
                split = os.path.splitext(args['config'])
                if(split[1]=='.py'): #rename python files' extension to make their icons change in VSCode
                    os.system(f"cp {args['config']:s} {outdir:s}/{split[0]:s}.{EXT}")
                else:
                    os.system(f"cp {args['config']:s} {outdir:s}/{args['config']:s}")
                '''

                #create just on jobscript for this config
                jobscript_name,njobs = create_jobscript_and_config(outdir, item["name"], script_basename, base_config, counter,
                                                             repeats=base_config.get("REPEATS",None), kfolds=config["CROSS_VALIDATION"]['K'] if use_kfcv else None)
                scripts_to_run.append(jobscript_name)
                jobcounter += njobs
                counter += base_config.get("REPEATS", None) or 1
                
            with open(os.path.join(outdir, "jobs.lst"), 'w') as f:
                f.write("\n".join(scripts_to_run))
            print(f"Created {jobcounter} jobs")
        else:
            configs.expand_config_dict(config, args["config"])
            config["RUN_ID"] = RUN_ID
            if("GRID" in config):
                del config["GRID"]

            _, njobs = create_jobscript_and_config(outdir, "", script_basename, config, None, repeats=config.get("REPEATS", None),
                                        kfolds=config["CROSS_VALIDATION"]['K'] if use_kfcv else None)
            print(f"Created {njobs} job(s)")
    elif(args["command"]=="run"):
        scripts_to_run = []
        counter = 0
        outdir = os.path.join(args["output_dir"], args["dir"])
        jobs_file = os.path.join(outdir, "jobs.lst")
        if(os.path.isfile(jobs_file)):
            with open(jobs_file, 'r') as f:
                scripts_to_run = [script.replace("\n", "") for script in f.readlines()]
            
            for script in scripts_to_run:
                print(f"Executing jobscript {counter:d} - {script}")
                os.system(f"sbatch {script}")
                counter += 1
        else:
            job_file = os.path.join(outdir, "jobscript.sh")
            if(os.path.exists(job_file)):
                print(f"Executing jobscript {job_file}")
                if(args["no_sbatch"]):
                    os.system(f"sh {job_file}")
                else:
                    os.system(f"sbatch {job_file}")
            else:
                print("No executable jobscript found.")