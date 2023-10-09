from typing import Callable

def extract_params_from_summary(model, lines:int=5) -> list:
    buf = []
    def print_fn(str):
        buf.append(str)
        if(len(buf)>lines):
            buf.pop(0)
    model.summary(print_fn=print_fn)
    return buf

def format_floats(d:dict|list, format_fn:Callable):
    for key, value in d.items():
        if(isinstance(value, (list,tuple,dict))):
            format_floats(value, format_fn)
        else:
            try:
                if(int(value)!=value):
                    d[key] = float(format_fn(value))
            except:
                pass