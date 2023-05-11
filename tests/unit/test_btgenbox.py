"""Tests the concreate implementarion of BtParser for Genbox-type Backtests"""

# Standard library imports
from pathlib import Path
from six import string_types

# Non-standard library imports
import pytest
import pandas as pd

# Project imports
from src.parser import (
    BT_FILES,
    __version__,
)
from src.parser.btparser import (
    EXTENSION_SEP,
    BtPeriods, 
    BtPlatforms,
    BtOrderType,   
)

from src.parser.btgenbox import (
    OPS_FINAL_COLUMN_NAMES,
    BtGenbox,
)


# Dummy test - Courtesy of JJ
def test_btgenbox_import_works_properly():
    pass
    

@pytest.fixture(autouse=True)
def bt():    
    return BtGenbox(Path(r'src/payload/'), BT_FILES[0])


class TestBtGenboxProperties:
    """Test expected values returned from accesing BtGenbox Properties."""
    
    def test_btgenbox_ordertype_returns_valid_type(self, bt):
        """ordertype properties should return a BtOrderType type."""
        assert isinstance(bt.ordertype, BtOrderType)
        
    def test_btgenbox_ordertype(self, bt):
        """ordertype properties should return a member of the BtOrderType Enum."""
        assert bt.ordertype == BtOrderType.BUY
    
    def test_btgenbox_name_returns_valid_type(self, bt):
        """name property should return a string type."""
        assert isinstance(bt.name, string_types)
    
    def test_btgenbox_name(self, bt):
        """name property should return a valid name for the backtest."""
        expected = BT_FILES[0].split(EXTENSION_SEP[0])[0]
        assert bt.name == expected
        
    def test_btgenbox_platform_returns_valid_type(self, bt):
        """platform property should return a member of the BtPlatform Enu."""
        assert isinstance(bt.platform, BtPlatforms)
        
    def test_btgenbox_platform(self, bt):
        """platform property should return a valid platform enum member."""
        expected = BtPlatforms.GBX
        assert bt.platform == expected
        
    def test_btgenbox_operations_returns_valid_type(self, bt):
        """operations property should return a DataFrame type."""
        assert isinstance(bt.operations, pd.DataFrame)        
    
    def test_btgenbox_period(self, bt):
        """period property should return a valid period enum member."""
        expected = BtPeriods.ISOS
        assert bt.period == expected
    
    @pytest.mark.xfail(__version__ > '0.1.0', reason='Version 0.2.0 checks symbol exists')    
    def test_btgenbox_symbol_returns_valid_type(self, bt):
        """symbol property should return a string type."""
        assert isinstance(bt.symbol, string_types)
    
    @pytest.mark.xfail(__version__ > '0.1.0', reason='Version 0.2.0 checks symbol exists')    
    def test_btgenbox_symbol(self, bt):
        """symbol property should return a correct symbol."""
        expected = 'AUDUSD'
        assert bt.symbol == expected
        
    def test_btgenbox_timeframe_returns_valid_type(self, bt):
        """timeframe property should return a string type."""
        assert isinstance(bt.timeframe, string_types)        
    
    @pytest.mark.xfail(__version__ < '0.2.0', reason='Not implementented until version 0.2.0') 
    def test_btgenbox_timeframe(self, bt):
        """timeframe property should return a valid period enum member."""
        expected = 'H4'
        assert bt.timeframe == expected

class TestOperationsDataFrameIntegrity:
    """Test integrity int he operations parsed by BtGenbox."""
    def test_operations_dataframe_with_correct_columns_names(self, bt):
        """operations datagframe should have the same number of columns as the reference."""
        expected = len(OPS_FINAL_COLUMN_NAMES)
        assert len(bt.operations.columns) == expected         
       
    def test_operations_dataframe_with_correct_columns_names(self, bt):
        expected = OPS_FINAL_COLUMN_NAMES         
        assert list(bt.operations.columns) == expected
                     
        
def test_btgenbox_init_contructs_correctly(bt):
    assert bt.path == Path(r'src/payload/')
    assert bt.file == BT_FILES[0]
    assert bt.platform == BtPlatforms.GBX
    assert bt.period == BtPeriods.ISOS
    