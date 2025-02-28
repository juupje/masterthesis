from typing import Callable
import json
class Encoder(json.JSONEncoder):
    def default(self,obj):
        import numpy
        if isinstance(obj,numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(Encoder, self).default(obj)

def extract_params_from_summary(model, lines:int=3) -> list:
    buf = []
    def print_fn(str:str, line_break=None):
        buf.extend(str.strip().split("\n"))
        while(len(buf)>lines):
            buf.pop(0)
    model.summary(print_fn=print_fn)
    return buf

def format_floats(d:dict|list, format_fn:Callable):
    for key, value in (d.items() if isinstance(d, dict) else zip(range(len(d)), d)):
        if(isinstance(value, (list,tuple,dict))):
            format_floats(value, format_fn)
        else:
            try:
                if(int(value)!=value):
                    d[key] = float(format_fn(value))
            except:
                pass