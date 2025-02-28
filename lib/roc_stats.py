#!/usr/bin/env python3
import steganologger
import yaml, os, re
import argparse
from pathlib import Path

def scan_dir(path):
    files = []
    func = Path(path).rglob if args.recursive else Path(path).glob
    for ext in args.extensions:
        files.extend(func(f"*.{ext:s}"))
    return files

'''def extract_info(data:dict|list, keys:str=""):
    l = {}
    if(type(data) is dict):
        for key, val in data.items():
            if(isinstance(val, (dict,list))):
                l.update(extract_info(val, f"{keys}~{key}"))
            elif(re.search(r"eps_[bs]\@eps_[bs]", key) is not None):
                l[f"{keys}~{key}"] = val
    elif(type(data) is list):
        for i in range(len(data)):
            if(isinstance(data[i], (dict,list))):
                l.update(extract_info(data[i], f"{keys}~{i:d}"))
    return l

def parse_info(info:dict):
    result = {}
    for key, val in info.items():
        keys = key[1:].split("~")
        print(keys)
        d = result
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = val
    return result'''
def extract_info(data:dict|list, keys:str=""):
    copy = data.copy()
    if(type(data) is dict):
        for key, val in copy.items():
            if(isinstance(val, (dict,list))):
                extract_info(val)
                if(len(val)==0):
                    del data[key]
            elif(type(key) is not str or re.search(r"eps_[bs]\@eps_[bs]", key) is None):
                del data[key]
    elif(type(data) is list):
        for item in copy:
            if(isinstance(item, (dict,list))):
                extract_info(item)
                if(len(item) == 0):
                    data.remove(item)
            else:
                data.remove(item)

if __name__=="__main__":
    parser = argparse.ArgumentParser("Lists stats of roc-curve plots")
    parser.add_argument("--recursive", "-r", help="Scan folders recursively", action='store_true')
    parser.add_argument("--extensions", "-e", help="File extensions to include", nargs='*', default=['pdf', 'svg', 'png'])
    parser.add_argument("files", help="Files to include (directories will be iterated over)", nargs='*')
    args = parser.parse_args()
    files = []
    for file in args.files:
        if(os.path.isdir(file)):
            files.extend(scan_dir(file))
        elif(os.path.isfile(file)):
            files.append(file)
        else:
            print(f"File not found: '{file}'")
    print(f"Scanning {len(files):d} files")

    for file in files:
        try:
            data, _, _ = steganologger.decode(file)
            extract_info(data)
            if(len(data)==0): continue
            print(str(file) + ":")
            print(yaml.safe_dump(data))
        except ValueError as e:
            print("\t", e)