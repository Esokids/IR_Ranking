import glob
from pathlib import Path

def open_read_file():
    file = list()
    doc = list()
    for filename in glob.glob('.\\File\\*.txt'):
        doc.append(Path(filename).stem)
        with open(filename, 'r') as f:
            file.append(f.read())
    return file, doc
