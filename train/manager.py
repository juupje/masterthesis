#!/usr/bin/env python3
import os
import argparse
from utils import configs, misc
import re
from datetime import datetime
import subprocess
import json

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

def guess_files(args, which=['jobscript', 'config', 'plot', 'script']):
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

def create_jobscript_and_config(outdir:str, item_dir:str, script_name:str, configuration:dict, counter:int) -> str:
    #create the output for this grid item
    fulldir = os.path.join(outdir, item_dir)
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    
    #write config to a file
    with open(os.path.join(fulldir, f"config.json"), "w") as f:
        json.dump(configuration, f, ensure_ascii=True, indent=2)
    #change jobscript
    with open(args["jobscript"], 'r') as f:
        jobscript = f.read()
        jobscript = workdir_seq.sub(f"WORKDIR={outdir:s}", jobscript) #directory with the main python script
        jobscript = rundir_seq.sub(f"RUNDIR=\"{item_dir:s}\"", jobscript) #directory with the config file (also the output directory)
        jobscript = config_seq.sub(f"CONFIG=\"$WORKDIR/$RUNDIR/config.json\"", jobscript)
        if(args["plot"]):
            jobscript = plot_seq.sub(f"PLOT_SCRIPT={os.path.splitext(args['plot'])[0]:s}.{EXT}", jobscript)
        else:
            jobscript = plot_seq.sub(f"PLOT_SCRIPT=", jobscript)
            jobscript = re.sub('-plot_script=[^\s]+', "", jobscript)
        jobscript = scriptname_seq.sub(f"SCRIPTNAME={script_name:s}.{EXT:s}", jobscript)
        jobscript = re.sub("--job-name=.*$",
                        f"--job-name={configuration['JOBNAME']:s}_{configuration['TAG']:s}_R{RUN_ID:d}" +\
                                    (f"-{counter:d}" if counter is not None else ""),
                        jobscript, 1, re.MULTILINE)
    #write jobscript to file
    jobscript_name = os.path.join(fulldir, "jobscript.sh")
    with open(jobscript_name, 'w') as f:
        f.write(jobscript)
    return jobscript_name

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Batch job manager")

    sp = parser.add_subparsers(help="commands", dest="command")

    p1 = sp.add_parser("create")

    p1.add_argument("model-dir", help="Model directory", type=str)
    p1.add_argument("--jobscript", "-j", help="job script name (in model directory)", type=str, required=False)
    p1.add_argument("--script","-s", help="script name (in model directory)", type=str, required=False)
    p1.add_argument("--config", "-c", help="configuration script name (in model directory)", type=str, required=False)
    p1.add_argument("--plot", "-p", help="plotting script name (in model directory)", type=str, required=False)
    p1.add_argument("--no-grid", "-n", help="Create a single job, no grid", action='store_true')
    p1.add_argument("--run-id", "-r", help="Use this specific run id", type=int)
    p1.add_argument("--extra-files", "-f", help="Extra files to be copied", type=str, nargs='*', default=[])
    p1.add_argument("--old-format", "-o", help="Use legacy format for backwards compatibility", action='store_true')

    p2 = sp.add_parser("run")
    p2.add_argument("dir", help="directory to run (relative to the outputs folder)", type=str)
    p2.add_argument("--output-dir", help="output directory, of which `dir` is a subdirectory (default: $HOME/thesis/outputs)", type=str, default=os.path.join(os.getenv("HOME"), "thesis", "outputs"))
    p2.add_argument("--no-sbatch", "-s", help="Run the jobscript with `sh`. Only applicable to single jobs", action='store_true')

    args = vars(parser.parse_args())

    if(args["command"]=="create"):
        os.chdir(args["model-dir"])
        #see if we need to guess the scripts
        scripts = ['jobscript', 'config', 'plot', 'script']
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
        
        workdir_seq = re.compile(r"^WORKDIR\s*=\s*.*$", re.MULTILINE) #for the dir in which the config script is
        rundir_seq = re.compile(r"^RUNDIR\s*=\s*.*$", re.MULTILINE) #for the dir in which the main script is
        config_seq = re.compile(r"^CONFIG\s*=\s*.*$", re.MULTILINE) #for the config file
        plot_seq = re.compile(r"^PLOT_SCRIPT\s*=\s*.*$", re.MULTILINE) #for the plot script
        scriptname_seq = re.compile(r"^SCRIPTNAME\s*=\s*.*$", re.MULTILINE) #for the main script
        if(use_grid):
            from copy import deepcopy
            scripts_to_run = []
            counter = 0
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

                if(base_config.get("REPEATS", None)==None):
                    #create just on jobscript for this config
                    jobscript_name = create_jobscript_and_config(outdir, item["name"], script_basename, base_config, counter)
                    scripts_to_run.append(jobscript_name)
                    counter += 1
                else:
                    for repeat in range(base_config["REPEATS"]):
                        #create multiple jobscripts with the same config
                        #increase the seed (repeats should never have the same seed)
                        base_config["SEED"] += 1
                        jobscript_name = create_jobscript_and_config(outdir, os.path.join(item["name"], str(repeat)),\
                                                                     script_basename, base_config, counter)
                        #store jobscript file name
                        scripts_to_run.append(jobscript_name)
                        counter += 1
                
            with open(os.path.join(outdir, "jobs.lst"), 'w') as f:
                f.write("\n".join(scripts_to_run))
            print(f"Created {counter} jobs")
        else:
            configs.expand_config_dict(config, args["config"])
            config["RUN_ID"] = RUN_ID
            if("GRID" in config):
                del config["GRID"]

            if(config.get("REPEATS", None)==None):
                create_jobscript_and_config(outdir, "", script_basename, config, None)
            else:
                jobscripts = []
                for repeat in range(config["REPEATS"]):
                    config["SEED"] += 1
                    jobscripts.append(create_jobscript_and_config(outdir, str(repeat), script_basename, config, repeat))
                with open(os.path.join(outdir, "jobs.lst"), 'w') as f:
                    f.write("\n".join(jobscripts))
                print(f"Created {len(jobscripts):d} jobs")
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