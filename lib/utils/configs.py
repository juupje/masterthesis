import os,json
from copy import deepcopy
import re
from .imports import import_mod

def marshal(dict_from:dict, dict_into:dict, name:str=None, quiet:bool=False) -> dict:
    text = []
    if(dict_from is None):
        if not quiet: text.append("input is none, using all defaults")
        result = deepcopy(dict_into)
    else:
        #copy all the keys which are not overwritten
        result = {key:deepcopy(val) for key,val in dict_into.items() if key not in dict_from}
        if not quiet: text.extend([f"Key '{key} not given, using default '{val}'" for key,val in result.items()])
        #copy all the keys that are overwritten and well as new keys
        for key, val in dict_from.items():
            result[key] = deepcopy(val)
    if(len(text)>0):
        print(f"Marshalling {name}:\n\t"+"\n\t".join(text))
    return result

def expand_config_dict(config:dict|list, config_file:str, _copy:dict|list=None, extra_dict:dict=None):
    if(_copy is None):
        _copy = deepcopy(config)
        if(extra_dict):
            _copy.update(extra_dict)
    if(type(config) is list):
        iterator = enumerate(config)
    else:
        iterator = iter(config.items())
    for key,value in iterator:
        if(isinstance(value, (list, dict))):
            expand_config_dict(value, config_file, _copy)
        if(type(value) is str):
            match = re.match(r"\$<\|((?:(?!<\|).)+)\|>", value)
            if(match):
                print("Expanding key " + key)
                config[key] = eval(match.group(1), globals(), _copy)
            else:
                match = re.search(r"\$<\|((?:(?!<\|).)+)\|>", value)
                if(match):
                    print("Expanding key " + key)
                    config[key] = value[:match.span(0)[0]] + str(eval(match.group(1),globals(),_copy))+value[match.span(0)[1]:]
    if("EXTRA_CONFIGS" in config):
        extra_configs = config["EXTRA_CONFIGS"]
        if(type(extra_configs) is not list): extra_configs = [extra_configs]
        for name in extra_configs:
            extra_config = parse_config(os.path.join(os.path.dirname(config_file), name), extra_dict=_copy)
            config.update(extra_config)
        del config["EXTRA_CONFIGS"]

def parse_config(config_file:str, expand=True, extra_dict=None) -> dict:
    if(os.path.isfile(config_file)):
        config = None
        if(config_file.endswith(".json")):
            with open(config_file, 'r') as f:
                config = json.load(f)
        elif(config_file.endswith(".py")):
            pat = re.compile("^[A-Z][A-Z0-9_]*$")
            config = {key:value for key,value in import_mod("config", config_file).__dict__.items() if pat.fullmatch(key)}
        else:
            raise ValueError("Unsupported script type: " + os.path.splitext(config_file)[1])
        if(expand):
            expand_config_dict(config, config_file, extra_dict=extra_dict)
        return config
    else: 
        raise ValueError("Config script " + config_file + " does not exist.")