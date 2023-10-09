from PIL import Image
import json
import os,sys
import argparse
CHECKKEYV1 = "01010110" #V1
CHECKKEYV2 = "01010111" #V2
SVG_KEY = "STEGANOLOGGER ENCODED DATA"
PDF_KEY = "/STEGANOLOGGER"
VERSION = 2
ENCODINGS = {"json": 1, "yaml": 2, "text": 3}

try:
    import numpy
    class Encoder(json.JSONEncoder):
        def default(self,obj):
            if isinstance(obj,numpy.integer):
                return int(obj)
            elif isinstance(obj, numpy.floating):
                return float(obj)
            elif isinstance(obj, numpy.ndarray):
                return obj.tolist()
            else:
                return super(Encoder, self).default(obj)
except:
    Encoder = json.JSONEncoder

def _convert_nans_and_infs(encoded_json):
    import re
    encoded_json = re.sub(r":\sInfinity,", ": \"Infinity\",", encoded_json)
    return re.sub(r":\sNaN,", ": \"NaN\",", encoded_json)

def _modify_pixels(pix, data, datatype):
 
    datalist = [f"{ord(x):08b}" for x in data]
    datalist = [CHECKKEYV2,f"{VERSION:08b}", f"{ENCODINGS.get(datatype,0):08b}"] + datalist
    lendata = len(datalist)
    imdata = iter(pix)
 
    for i in range(lendata):
 
        # Extracting 3 pixels at a time
        pix = list(imdata.__next__()[:3]+ imdata.__next__()[:3]+ imdata.__next__()[:3])
 
        # Pixel value should be odd for 1 and even for 0
        for j in range(8):
            if(int(datalist[i][j]) != pix[j]%2):
                pix[j] += -1 if pix[j] > 0 else 1
        # continue bit: 0 means keep reading; 1 means stop
        continue_bit = 1 if i == lendata-1 else 0
        if(pix[-1]%2 != continue_bit):
            pix[-1] += -1 if pix[-1]>0 else 1
       
        pix = tuple(pix)
        yield pix[0:3]
        yield pix[3:6]
        yield pix[6:9]

def image_encode(file:str, data:str|dict, datatype:str,to_file:str):
    if(type(data) is dict):
        data = json.dumps(data, ensure_ascii=True, cls=Encoder)
    from PIL import Image
    img = Image.open(file)
    img_copy = img.copy()
    img.close()
    w = img_copy.size[0]
    (x, y) = (0, 0)
    count = 0
    for pixel in _modify_pixels(img_copy.getdata(), data, datatype):
        # Putting modified pixels in the new image
        img_copy.putpixel((x, y), pixel)
        count += 1
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1
    print(f"Encoded {(count//3)} bytes in {count} pixels ({count/(img_copy.size[0]*img_copy.size[1])*100:.2f}% of image)")

    img_copy.save(to_file, "PNG")

def pgf_encode(file:str, data:str|dict, datatype:str,to_file:str):
    if(file == to_file):
        out_file = open(file, 'a')
    else:
        out_file = open(to_file, 'w')
        with open(file, 'r') as f:
            for line in f:
                out_file.write(line)
    out_file.write("%% " + SVG_KEY + " START\n")
    out_file.write("%% "+json.dumps({"data": data, "datatype": datatype, "version": VERSION}, indent=2).replace("\n", "\n%%") + "\n")
    out_file.write("%% "+ SVG_KEY + " END\n")
    print("Encoded data successfully as PGF comment!")
    out_file.close()

def pdf_encode(file:str, data:str|dict, datatype:str,to_file:str):
    from PyPDF2 import PdfFileReader, PdfFileMerger
    file_in = open(file, 'rb')
    pdf_reader = PdfFileReader(file_in)
    metadata = pdf_reader.getDocumentInfo()
    print(metadata)

    pdf_merger = PdfFileMerger()
    pdf_merger.append(file_in)
    pdf_merger.addMetadata({
        PDF_KEY: json.dumps({"datatype": datatype, "version": VERSION, "data": data})
    })
    file_out = open(to_file, 'wb')
    pdf_merger.write(file_out)
    print("Encoded data successfully into PDF metadata!")
    file_in.close()
    file_out.close()

def svg_encode(file:str, data:str|dict, datatype:str, to_file:str):
    if(file == to_file):
        out_file = open(file, 'a')
    else:
        out_file = open(to_file, 'w')
        with open(file, 'r') as f:
            for line in f:
                out_file.write(line)
    out_file.write("<!-- " + SVG_KEY + "\n")
    out_file.write(json.dumps({"data": data, "datatype": datatype, "version": VERSION}, indent=2) + "\n")
    out_file.write(SVG_KEY +" -->\n")
    print("Encoded data successfully as SVG comment!")
    out_file.close()

def encode(file_path:str, data:dict|str, datatype:str=None,overwrite:bool=False, to_file:str=None) -> bool:
    """
    Parameters
    ----------
    file_path: str,
        Path of the image in which to encode the data
    data: dict,
        The dictionary to encode
    datatype: str,
        One of 'json', 'yaml', 'text'. This specifies the type of data you are trying to encode.
        If not given, the encoder will automatically check if it happens to be json. Otherwise, the type will be 'text'.
        This has no influence on how the data is encoded, but might be helpful for the decoder.
    overwrite: bool,
        If `True`, overwrite the original file with the encoded file. 
        f the original file did not have the .png extension, it is added automatically
    to_file: str,
        If `overwrite` is `True`, this parameter can be used to determine the file to which
        the encoded image is saved. If it is not given, the string `-encoded` will be appended to 
        the original image's name.
    """
    try:
        print("STEGANOLOGGER: encoding...")
        if(type(data) is dict):
            data = json.dumps(data, ensure_ascii=True, cls=Encoder)
            data = _convert_nans_and_infs(data)
            datatype = 'json'
        if(type(data) is not str):
            ValueError("Cannot encode this type of data")

        #if the datatype is none try to find out what it is
        if(datatype is None):
            try:
                json.loads(data)
                datatype = 'json'
                print("Determined that datatype is json")
            except json.JSONDecodeError as e:
                print("Could not auto-determine datatype, going with 'text'")
                datatype = 'text'

        if(not os.path.isfile(file_path)):
            print("Given image file does not exist.", file=sys.stderr)
        ext = os.path.splitext(file_path)[1].lower()

        if(overwrite):
            to_file = file_path
        else:
            if(to_file is not None):
                if(os.path.splitext(to_file)[1] != ext):
                    to_file += ext
            else:
                to_file = os.path.splitext(file_path)[0]+"-encoded" + ext
        
        if(ext == ".png"):
            image_encode(file_path, data, datatype, to_file)
        elif(ext == ".pdf"):
            pdf_encode(file_path, data, datatype, to_file)
        elif(ext == ".svg"):
            svg_encode(file_path, data, datatype, to_file)
        elif(ext == ".pgf"):
            pgf_encode(file_path, data, datatype, to_file)
        else:
            print("Extension " + ext + " is not supported.")
        print("Done!\n")
    except Exception as e:
        print("Something went wrong, could not encode data!", file=sys.stderr)
        print(e)

def png_decode(image_path):
    def _decode_pixels(pixels):
        # string of binary data
        assert len(pixels)==9
        binstr = ''.join(str(i%2) for i in pixels)
        return binstr

    def getbyte():
        return [value for value in imgdata.__next__()[:3] +
                                imgdata.__next__()[:3] +
                                imgdata.__next__()[:3]]

    if(not os.path.isfile(image_path)):
        print("No such file")
        exit()
    image = Image.open(image_path, 'r')
 
    data = ''
    imgdata = iter(image.getdata())
    binstr = _decode_pixels(getbyte())
    version = 1
    type_str = None
    if(binstr[:8] == CHECKKEYV2):
        # got version >2
        version = int(_decode_pixels(getbyte())[:8],2)
        type_id = int(_decode_pixels(getbyte())[:8],2)
        for key, value in ENCODINGS.items():
            if(value==type_id):
                type_str = key
    elif(binstr[:8] == CHECKKEYV1):
        type_id = None
    else:
        print("Check key does not match: no encoded data found. Got: " + binstr[:8])
        exit()

    while (binstr[-1]=='0'):
        binstr = _decode_pixels(getbyte())
        data += chr(int(binstr[:8],2))
    return data, version, type_str

def pgf_decode(svg_path):
    import re
    start = re.compile(r"%%\s*"+SVG_KEY+r"\sSTART\s*$")
    end = re.compile(r"%%\s*"+SVG_KEY+r"\sEND\s*$")
    with open (svg_path, 'r') as f:
        reading = False
        lines = ""
        for line in f:
            if(reading):
                if(end.match(line)):
                    break
                lines += line[2:] #remove the %%
            if(start.match(line)):
                reading = True
        if(len(lines)>0):
            doc = json.loads(lines)
            return doc["data"], doc["version"], doc["datatype"]
        else:
            print("No encoded information found")
            exit()

def pdf_decode(pdf_path):
    from PyPDF2 import PdfFileReader
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfFileReader(file)
        metadata = pdf_reader.getDocumentInfo()
        
        if(PDF_KEY in metadata):
            doc = json.loads(metadata["/EncodedInfo"])
            return doc["data"], doc["version"], doc["datatype"]
        else:
            print("No encoded data found!")
            exit()

def svg_decode(svg_path):
    import re
    start = re.compile(r"<!--\s*"+SVG_KEY+r"\s*$")
    end = re.compile(SVG_KEY + r"\s*-->$")
    with open (svg_path, 'r') as f:
        reading = False
        lines = ""
        for line in f:
            if(reading):
                if(end.match(line)):
                    break
                lines += line
            if(start.match(line)):
                reading = True
        if(len(lines)>0):
            doc = json.loads(lines)
            return doc["data"], doc["version"], doc["datatype"]
        else:
            print("No encoded information found")
            exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stenagologger')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encode", "-e", help="Run in encoding mode", action='store_true')
    group.add_argument("--decode", "-d", help="Run in decoding mode", action='store_true')

    parser.add_argument("--data", help="Data to encode in JSON format. Use @ to reference a file", type=str)
    parser.add_argument("--file", help="The file to decode/encode", type=str, required=True)
    parser.add_argument("--out-file", help="If encoding, store the result to this file", type=str)
    parser.add_argument("--pretty", help="If decoding, output in pretty json", action="store_true")

    args = vars(parser.parse_args())
    for s in ["data", "out_file", "file"]:
        if(s in "args" and args[s].startsWith("~")): args[s] = os.path.expanduser(args[s])
    if(args["encode"]):
        if("out_file" not in args or not args["out_file"]):
            print("No output file given")
            exit()
        if("data" not in args or not args["data"]):
            print("No data given")
            exit()
        data = args["data"]
        try:
            if(data[0]=='@'):
                with open(data[1:], 'r') as f:
                    data = json.load(f)
            else:
                data = json.loads(data)
        except json.JSONDecodeError as e:
            print(e)
        encode(args["file"], data, overwrite=False, to_file=args["out_file"])
    elif(args["decode"]):
        pretty = args["pretty"]
        ext = os.path.splitext(args["file"])[1]
        if(ext == ".png"):
            data, version, type_id = png_decode(args["file"])
        elif(ext == ".pdf"):
            data, version, type_id = pdf_decode(args["file"])
        elif(ext == ".svg"):
            data, version, type_id = svg_decode(args["file"])
        else:
            print("Extension " + ext + " is not supported.")

        if(type_id == 'json'):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                print(e)
                print("Found type 'json', but could not interpret decoded text as json")
                type_id = 'text'
        elif(type_id == 'yaml'):
            import yaml
            try:
                data = yaml.safe_load(data)
            except Exception as e:
                print(e)
                print("Found type 'yaml', but could not interpret decoded text as yaml")
                type_id = 'text'
        elif(type_id is None):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                print("Failed to interpret data as json")
                type_id = 'text'
        
        print("Parsing data as " + type_id)
        if("out_file" in args and args["out_file"]):
            with open(args["out_file"], 'w') as f:
                if(type_id=='json'):
                    json.dump(data,f, indent=2 if pretty else None)
                elif(type_id=='yaml'):
                    yaml.dump(data,f,indent=2)
                else:
                    f.write(data)
        else:
            if(type_id=='json'):
                print(json.dumps(data, indent=2 if pretty else None))
            elif(type_id=='yaml'):
                print(yaml.dump(data,indent=2))
            else:
                print(data)
    else:
        print("Nothing to do")