import re
import numpy as np
import numpy.typing as npt
import h5py

class Discriminator:
    """
    Selects data based on a key and a range. Useful for selecting events based on `Mjj` or `Pt` for example.
    """
    def __init__(self, key:str, lower:float, upper:float, col_idx:int=None):
        if(col_idx is None):
            match = re.search(r"\/(-?\d+)$", key)
            if(match):
                self.key = key[:match.start()]
                self.col_idx = int(match.group(1))
            else:
                self.key = key
                self.col_idx = None
        else:
            self.key = key
            self.col_idx = col_idx
        self.lower = lower
        self.upper = upper
    
    def apply(self, data:h5py.HLObject, _slice:slice=None) -> npt.NDArray[np.bool_]:
        """
        returns a boolean array of the same length as the data, with True for the events that are within the range.
        """
        if(self.col_idx):
            data = np.array(data[self.key][:,self.col_idx])
        else:
            data = np.array(data[self.key])
        if(_slice):
            data = data[_slice]
        assert len(data.shape)==1, "Discriminator has multiple values"
        idx = np.ones(data.shape[0],dtype=bool)
        if(self.lower):
            idx = np.logical_and(idx, data>=self.lower)
        if(self.upper):
            idx = np.logical_and(idx, data<self.upper)
        return idx
    def __str__(self) -> str:
        return f"Discriminator(key={self.key:s}, column={self.col_idx:d}, lower={self.lower:.3g}, upper={self.upper:.3g})"
