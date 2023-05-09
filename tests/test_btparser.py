# Standard library imports
"""Tests the Abstract Base class BtParser"""

import sys
from pathlib import Path
from unittest.mock import patch

# Non-standard library imports

# Projec imports
from ..src.parser.btparser import BtParser

sys.path.append('..')


# Following is a dummy test to check that the imports are working properly
# (not always the case, coming from JJ LoL)
def test_btparser_import_works_properly():
    """Dummy test: Test whether the project import works or not"""
    assert True == True
    

'''
@patch("..src.parser.btparser.BtParser.__abstractmethods__", set())
def test_btparser_init():
    """Need to patch the base class, since it is absstract"""
    bt = BtParser(Path('/'), 'example.txt')
    assert bt.path == Path('/')
    
    assert bt.file == 'example.txt'
'''    