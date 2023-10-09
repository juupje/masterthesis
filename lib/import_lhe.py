#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A python script which reads LHE (Les Houches Event) files

@author: Joep Geuskens
"""
import xml.etree.ElementTree as ElementTree
import numpy as np

"""
Particle status codes:
    -1 incoming
    1 outgoing final state
    -2 intermediate space like propagator (x and Q^2 conserved)
    2 intermediate resonance, Mass preserved
    3 intermediate resonance (only for documentation)
    -9 incoming beam particles at t=-infty

    23 pythia outgoing hard-process particle
    62 pythia outgoing subprocess particle
"""
multi_particles = {"q": [1,2,3,4], "q~": [-1,-2,-3,-4], "j":[1,2,3,4,21,-1,-2,-3,-4], "l-": [11,13], "l+": [-11,-13]}
particle_ids = [  1,   2,   3,   4,   5,   6,  11,   12,  13,  14,    15,  16,    21,  22,  23,  24,  25]
particle_names=["d", "u", "s", "c", "b", "t", "e", "en", "m", "mn", "ta", "tn", "g", "a", "z", "w", "h"]

def particle_name(pdgid):
    if(isinstance(pdgid, list)):
        return [particle_name(pdgid_i) for pdgid_i in pdgid]
    abspdgid = abs(pdgid)
    s = ""
    try:
        s = particle_names[particle_ids.index(abspdgid)]
    except ValueError:
        raise ValueError("Particle with id " + pdgid + "does not exist.")
    if(abspdgid==24 or abspdgid==11 or abspdgid==13 or abspdgid==15):
        return s + ("+" if pdgid<0 else "-")

    if(pdgid>0):
        return s
    elif(pdgid > -16):
        return s + "~"
    else:
        raise ValueError(s + " does not have an antiparticle")

def particle_id(name):
    if(isinstance(name,list)):
        return [particle_id(name_i) for name_i in name]
    name = name.strip()
    if(name in multi_particles):
        return multi_particles[name].copy()
    if(name[-1]=='+'):
        antiparticle=True
        name = name[:-1]
    elif(name[-1]=='-'):
        antiparticle=False
        name = name[:-1]    
    elif(name[-1]=="~"):
        antiparticle = True
        name = name[:-1]
    else:
        antiparticle = False
    try: 
        pdgid = particle_ids[particle_names.index(name)]
        if(pdgid>16 and pdgid!=24 and antiparticle):
            raise ValueError("particle " + name + " does not have an antiparticle")
        return pdgid if not antiparticle else -pdgid
    except ValueError:
        raise ValueError("Particle with name " + str(name) + " not found.")
    
class Data:
    def __init__(self):
        self.version = 3
        self.events = []
        
    def get_event(self, index):
        return self.events[index]
    
    def add_event(self, event):
        self.events.append(event)
        
    def get_particles(self, pdg_id):
        to_return = []
        for event in self.events:
            to_return.extend(event.get_particles(pdg_id))
        return to_return
        
    def as_numpy_array(self):
        arr = np.empty((len(self.events)), dtype=list)
        for i in range(len(self.events)):
            arr[i] = self.events[i].as_list()
        return arr
    
    @property
    def n_events(self):
        return len(self.events)

class Event:
    def __init__(self, n_particles, particles=None):
        self._n_particles = n_particles
        if(particles is None):
            self._particles = []
        else:
            self._particles = particles
            if(len(particles) > n_particles):
                raise IndexError("There were {:d} particles given, but the event size is {:d}".format(len(particles), n_particles))
    
    @property
    def n_particles(self):
        return self._n_particles
    
    @n_particles.setter
    def n_particles(self, new_val):
        self._n_particles = new_val
    
    @property
    def is_filled(self):
        return self._n_particles == len(self._particles)
    
    @property
    def sqrt_s(self):
        s = np.sum([p_i.E for p_i in self._particles if p_i._status==1])
        return s
    
    def __add_particle__(self, particle):
        if(len(self._particles)<self._n_particles):
            self._particles.append(particle)
        else:
            raise IndexError("This event is already filled.")
    
    def get_particles(self, pdg_id):
        to_return = []
        if(isinstance(pdg_id, list)):
           for pdgid in pdg_id:
               to_return.extend(self.get_particles(pdgid))
        else:
            for particle in self._particles:
                if(particle.pdgid==pdg_id):
                    to_return.append(particle)
        return to_return
    
    def get_final_particles(self):
        return [p for p in self._particles if (p._status==1 or p._status==23 or p._status==62)]
    
    def as_list(self):
        l = []
        for p in self._particles:
            l.append(p.as_numpy_array())
        return l
    
    def __str__(self):
        return "LHE Event with {:d} particles: ".format(self._n_particles) + ", ".join(particle_name([int(p.pdgid) for p in self._particles]))

class Particle:
    def __init__(self, pdg_id, px, py, pz, E, m, s, status):
        self._pdg_id = int(pdg_id)
        self._px = px
        self._py = py
        self._pz = pz
        self._E = E
        self._m = m
        self._s = s
        self._status = int(status)

    @property
    def p(self):
        #we make a copy, to ensure that no outside operation can influence the properties
        return np.array([self._E,self._px,self._py,self._pz])

    @property
    def p3(self):
        return (self._px, self._py, self._pz)
    
    @property
    def eta(self):
        """
        Returns the pseudorapidy.
        Copied from the ROOT documententation
        """
        ct = self.costheta
        if(ct*ct<1): return -0.5*np.log((1.0-ct)/(1.0+ct))
        if(self._pz==0): return 0
        if(self._pz>0): return 10e10
        else: return -10e10
    
    @property
    def phi(self):
        if(self._px==0 and self._py==0 and self._pz==0):
            return 0
        else:
            return np.arctan2(self._py, self._px)
    
    @property
    def costheta(self):
        mag = np.sqrt(self._px**2+self._py**2+self._pz**2)
        if(mag==0):
            return 1.0
        else:
            return self._pz/mag
    
    @property
    def Et(self):
        pt2 = self._px**2+self._py**2
        Et2 = 0 if pt2==0 else (self._E**2)*pt2/(pt2+self._pz**2)
        return -np.sqrt(Et2) if self._E<0 else np.sqrt(Et2)
    
    @property
    def E(self):
        return self._E
    
    @property
    def m(self):
        return self._m
    
    @property
    def px(self):
        return self._px
    
    @property
    def py(self):
        return self._py
    
    @property
    def pz(self):
        return self._pz
    
    @property
    def pt(self):
        return np.sqrt(self._px**2+self._py**2)
    
    @property
    def spin(self):
        return self._s
    
    @property
    def pdgid(self):
        return self._pdg_id
    
    @property
    def status_code(self):
        return self._status
    
    def as_numpy_array(self):
        return np.array([self._pdg_id, self._px, self._py, self._pz, self._E, self._m, self._s, self._status], dtype=np.float32)

    def __str__(self):
        return "{:s} p=({:.3g}, {:.3g}, {:.3g}, {:.3g}) m={:.3g}, s={:.1f}, status={:d}".format(particle_name(self._pdg_id), self._px, self._py, self._pz, self._E, self._m, self._s, self._status)
        
def readLHEData(fname, data=None, maxEvents=None):
    tree = ElementTree.parse(fname)
    tree_root = tree.getroot()
    version = tree_root.attrib["version"]
    if(version != "3.0"):
        print("Only version 3.0 is supported, got version " + version + " errors may occur.")
    if(data is None):
        data = Data() #create new Data object
    for child in tree_root:
        if(child.tag == "event"):
            lines = child.text.strip().split("\n")
            N = int(lines[0].split()[0].strip())
            event = Event(N)
            for i in range(1, N+1):
                particle_data = lines[i].strip().split()
                particle = Particle(int(particle_data[0]), float(particle_data[6]), float(particle_data[7]), float(particle_data[8]),
                                    float(particle_data[9]), float(particle_data[10]), float(particle_data[12]), int(particle_data[1]))
                event.__add_particle__(particle)
            data.add_event(event)
            if(maxEvents is not None and data.n_events==maxEvents):
                break
    return data

def from_numpy_array(arr):
    data = Data()
    for event in arr:
        particles = []
        for particle in event:
            if(particle is not None):
                particles.append(Particle(*particle))
        data.add_event(Event(len(particles), particles))
    return data