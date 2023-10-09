#! /usr/bin/env python3
import tqdm, ROOT
import numpy as np
import os, sys
import h5py
import pandas as pd
import argparse

mad_output = os.getenv("MAD_OUTPUT")
mad_dir = os.getenv("MADGRAPH_DIR")
branches = [""]
if(mad_dir is None):
    mad_dir = "."
if(mad_output is None):
    mad_output = "."
cwd = os.getcwd()

def initialiseROOT():
    os.chdir(f"{mad_dir}/Delphes/")
    ROOT.gSystem.Load(f"libDelphes.so")

    try:
        ROOT.gInterpreter.Declare(f'#include "classes/DelphesClasses.h"')
        ROOT.gInterpreter.Declare(
            f'#include "external/ExRootAnalysis/ExRootTreeReader.h"')
    except:
        pass


def generateListOfFiles(pathToEvents, start=0, end=-1):
    if os.path.isfile(pathToEvents):
        print(pathToEvents)
        return [pathToEvents]

    files = []
    for dName, sdName, fList in os.walk(pathToEvents):
        for filename in fList:
            if filename.endswith('.root'):
                files.append(os.path.join(pathToEvents, dName, filename))
    return sorted(files)


def getEvents(files, outputfile='/scratch/work/geuskens/tmp.h5', append=False, signal=None, cut:int=1200):
    def eventSelection(pt_min=cut):
        nJets = branchFatJet.GetEntriesFast()
        if nJets < 1: return False
        for i in range(nJets):
            if branchFatJet.At(i).PT > pt_min: return True

        return False


    def check_files(files, outputfile):
        isfile = [os.path.isfile(file) for file in files]
        assert all(isfile), \
            f'Files {np.array(files)[np.invert(isfile)]} not found'
        if(outputfile[0]!="/"):
            outputfile = f"{cwd}/{outputfile}"
        if(not(outputfile.endswith(".h5"))):
            outputfile += ".h5"
        return outputfile

    def openHDFfile(filename):
        newfile = False
        if(os.path.isfile(filename)):
            if(not(append)):
                os.remove(filename)
                newfile = True
        else:
            newfile = True
        file = pd.HDFStore(filename)
        colnames = [x for i in range(700) for x in [f"pt{i}", f"eta{i}", f"phi{i}"]]
        assert(len(colnames)==2100)
        if(signal is not None):
            colnames.append("signal")
        if(newfile):
            pd.DataFrame(dtype=np.float32, columns=colnames).to_hdf(file, "Particles",
                    format="table", complevel=1,complib="blosc:zlib")
        else:
            assert file.select("Particles", start=1,stop=1).columns == colnames, "Column names of existing file do not match"
        return file, colnames

    def writeToFile(events, file):
        if(signal is not None):
            assert columns[-1] == "signal"
            df = pd.DataFrame(events, dtype=np.float64, columns=columns[:-1])
            df["signal"] = signal
        else:
            df = pd.DataFrame(events, dtype=np.float64, columns=columns)
        df.to_hdf(file, "Particles",
            format="table", append="true",index=False, complevel=1, complib="blosc:zlib")

    print("Starting getEvents")
    outputfile = check_files(files, outputfile)
    output,columns = openHDFfile(outputfile)

    events = []
    # Loop over every input file given as an argument
    for file in files:
        print(file)
        # Create chain of root trees
        chain = ROOT.TChain("Delphes")
        chain.Add(file)


        # Create object of class ExRootTreeReader
        treeReader = ROOT.ExRootTreeReader(chain)
        numberOfEntries = treeReader.GetEntries()

        print("Chain contains %i Entries" % numberOfEntries)

        # Branches needed for reconstruction of the constituents
        branchEFlowTrack = treeReader.UseBranch("EFlowTrack")
        branchEFlowPhoton = treeReader.UseBranch("EFlowPhoton")
        branchEFlowNeutralHadron = treeReader.UseBranch("EFlowNeutralHadron")
        branchFatJet = treeReader.UseBranch("FatJet")


        # Loop over all events
        count = 0
        for entry in tqdm.tqdm(range(0, numberOfEntries)):
            treeReader.ReadEntry(entry)
            if not eventSelection(pt_min=cut):
                count += 1
                continue

            event_particles = []
            for i in range(branchEFlowTrack.GetEntriesFast()):
                particle = branchEFlowTrack.At(i)
                event_particles.append([particle.PT, particle.Eta, particle.Phi])

            for i in range(branchEFlowPhoton.GetEntriesFast()):
                particle = branchEFlowPhoton.At(i)
                event_particles.append([particle.ET, particle.Eta, particle.Phi])

            for i in range(branchEFlowNeutralHadron.GetEntriesFast()):
                particle = branchEFlowNeutralHadron.At(i)
                event_particles.append([particle.ET, particle.Eta, particle.Phi])

            tmp = None
            if(len(event_particles) <= 700):
                tmp = np.zeros(2100, dtype=np.float64)
                event_particles = np.array(event_particles, dtype=np.float64).flatten()
                tmp[:len(event_particles)] = event_particles
            else:
                event_particles = np.array(event_particles, dtype=np.float64)
                idx = np.arg_sort(event_particles[:,0])[-700:] #get the 700 particles with the largest pt
                np.random.shuffle(idx) #shuffle them, such that they are no longer ordered according to pt
                tmp = event_particles[idx].flatten()
            events.append(tmp)

            if(len(events)==500):
                writeToFile(events, output)
                events = []
        print(f"{count} events rejected")
    if(len(events)>0):
        writeToFile(events, output)
    #print(f"Saved dataframe of shape {output['Particles'].shape} to {outputfile}")
    print("Done")
    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts .root files to LHCO-2022 format")
    parser.add_argument("-o", "--output", help="Output file", type=str, required=True)
    parser.add_argument("-a", "--append", help="Append to the output file", action="store_true")
    parser.add_argument("--signal", help="Sets the signal bit", type=int)
    parser.add_argument("--cut", help="value of the fatjet cut", type=int, default=1200)
    group = parser.add_argument_group()
    parser.add_argument("-i", "--input", help="Initial arguments", nargs='+', type=str)
    group.add_argument("-d", "--dir", help="Madgraph output directory", type=str)
    group.add_argument("-r", "--run", help="name of the runs", nargs="+", type=str)
    group.add_argument("-t", "--tag", help="name of the tags, one per run", nargs="+", type=str)
    args = vars(parser.parse_args())
    # some extra processing of arguments
    if(not(args["dir"] is None) == (args["run"] is None) == (args["tag"] is None)):
        print("Arguments --dir, --run and --tag should be used together.")
        exit()
    if(args["run"] and len(args["run"]) != len(args["tag"])):
        if(len(args["tag"]) == 1):
            args["tag"] = args["tag"]*len(args["run"])
        elif(len(args["run"]) == 1):
            args["run"] = args["run"]*len(args["tag"])
        else:
            print("Number of runs and tags should be the same.")
            exit()
    if(args["input"] is None):
        if(args["dir"] is None):
            print("Argument --input or all of --dir, --run  and --tag are required")
            exit()
    else:
        if(args["dir"] is not None):
            print("Ignoring argument --dir and/or --run and --tag")
    if(args["signal"] is None):
        print("Warning: no signal bit given!")


    #and now on to the main program
    files = []
    if(args["input"]):
        for i in range(0,len(args["input"])):
            files += generateListOfFiles(args["input"][i])
    else:
        for run, tag in zip(args["run"], args["tag"]):
            dir = f"{mad_output}/{args['dir']}/Events/{run}"
            files += generateListOfFiles(f"{dir}/{tag}_delphes_events.root")
    initialiseROOT()
    getEvents(files, outputfile=args["output"], append=args["append"], signal=args["signal"], cut=args["cut"])
