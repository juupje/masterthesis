import re
import utils
import argparse, json, os
parser = argparse.ArgumentParser("Check config calls in a script")
parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", required=True, type=str)
parser.add_argument("--script", "-s", help="Path of the script", type=str)
parser.add_argument("--list", "-l", help="List contents of config file", action='store_true')
parser.add_argument("--name", "-n", help="Name to check for according to <name>['<key>']", type=str, default="config")
args = vars(parser.parse_args())

pat = re.compile(f"{args['name']}\[\"([A-Z][_A-Z]*)\"\]")
config = utils.parse_config(args["config"])

if(args["list"]):
    print(json.dumps(config, indent=True))

if(args["script"]):
    with open(args["script"]) as f:
        for line in f:
            for match in pat.findall(line):
                if match not in config:
                    print("Config key " + match + " could not be found")
