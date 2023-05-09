"""Tests the concretate implementarion of BtParser for Genbox-type Backtests"""

# Standard library imports
from pathlib import Path

# Non-standard library imports
import pytest

# Project imports
from src.parser import BT_FILES
from src.parser.btparser import (
    BtPeriods, 
    BtPlatforms,
    BtOrderType,   
)

from src.parser.btgenbox import BtGenbox


# Dummy test - Courtesy of JJ
def test_btgenbox_import_works_properly():
    assert True == True
    

@pytest.fixture()
def bt():    
    return BtGenbox(Path(r'src/payload/'), BT_FILES[0])


def test_btgenbox_ordertype_property_returns_valid_ordertype(bt):
    assert bt.ordertype == BtOrderType.BUY

    
def test_btgenbox_init_contructs_correctly(bt):
    assert bt.path == Path(r'src/payload/')
    assert bt.file == BT_FILES[0]
    assert bt.platform == BtPlatforms.GBX
    assert bt.period == BtPeriods.ISOS
    