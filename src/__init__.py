from os import listdir
from .parser.btparser import BT_EXTENSIONS
BT_FILES = [file for file in listdir(r'src/payload/') \
    if file.endswith(BT_EXTENSIONS)]