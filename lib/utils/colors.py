import os, json

class ColorIter:
    def __init__(self, which="big"):
        import json
        path = os.path.join(os.path.dirname(__file__),"colors.json")
        with open(path, 'r') as f:
            doc = json.load(f)
            if(which not in doc):
                raise ValueError(f"Unknown key: '{which}'")
            self.colors = [value for _, value in doc[which].items()]
        self.idx = 0
    def __iter__(self):
        return self
    def rewind(self):
        self.idx = 0
    def __next__(self):
        try:
            return self.colors[self.idx]
        except IndexError:
            raise StopIteration
        finally:
            self.idx += 1

def hex_to_rgb(hex):
    hex = hex.replace("#","")
    r,g,b = [int(hex[i*2:(i+1)*2],16) for i in range(3)]
    return r,g,b

def rgb_to_hex(rgb):
    def pad(x):
        if(len(x)==1):
            return "0"+x
        if(len(x)==0):
            return "00"
        return x
    return "#"+"".join([pad(hex(v)[2:]) for v in rgb])

def change_brightness(color, brightness):
    if type(color) is tuple:
        rgb = color[:3]
    elif type(color) is str:
        rgb = hex_to_rgb(color)
    if(type(brightness) is int):
        brightness /= 255
    m = max(rgb)
    rgb = [int(v*255/m*brightness) for v in rgb] #scale such that the max is 255, then scale according to brightness
    return rgb if type(color) is tuple else rgb_to_hex(rgb)


    