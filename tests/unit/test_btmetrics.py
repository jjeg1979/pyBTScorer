"""Tests the concreate implementarion of BtParser for Genbox-type Backtests"""

# Standard library imports
from pathlib import Path
from six import string_types
from decimal import Decimal
from datetime import timedelta

# Non-standard library imports
import pytest
import pandas as pd


# Project imports
from src.parser import (
    BT_FILES,
    __version__,
)

from src.parser.btgenbox import (   
    BtGenbox,
)

from src.parser.btmetrics import (
    ALL_METRICS,
    DEC_PREC,
    BtMetrics,
)

@pytest.fixture(autouse=True)
def mt():
    bt = BtGenbox(Path(r'src/payload'), BT_FILES[0])
    return BtMetrics(bt)

# Dummy test - Courtesy of JJ
def test_btmetrics_import_works_properly():
    assert True == True
    

@pytest.mark.metricsprops
class TestBtmetricsProperties:
    """Test expected values returned from accesing Btmetrics Properties."""
    
    def test_btmetrics_operations_returns_valid_type(self, mt):
        """operations properties should return a Pandas DataFrame."""
        assert isinstance(mt.operations, pd.DataFrame)   
        
    def test_btmetrics_available_metrics_returns_valid_type(self, mt):
        """available_metrics should return a set of strings."""
        assert isinstance(mt.available_metrics, set)
        for metric in mt.available_metrics:
            assert isinstance(metric, string_types)
        
    def test_btmetrics_available_metrics(self, mt):
        """available_metrics should return the correct names of the metrics"""
        assert mt.available_metrics == ALL_METRICS        


@pytest.mark.metricsvalues
class TestBtMetricsCalculations:
    """Test the correct calculations from the BTMetrics class"""
    def test_bt_metrics_calculate_pf_returns_a_decimal(self, mt):       
        assert isinstance(mt.calculate_pf(), Decimal)
    
    def test_bt_metrics_calculate_pf_in_pips(self, mt):
        expected = Decimal(1.84).quantize(Decimal(DEC_PREC))
        assert mt.calculate_pf()== expected
        
    def test_bt_metrics_calculate_pf_in_money(self, mt):
        expected = Decimal(1.84).quantize(Decimal(DEC_PREC))
        assert mt.calculate_pf(pips_mode=False) == expected
    
    def test_bt_metrics_drawdown_returns_a_series(self, mt):
        assert isinstance(mt.drawdown(), pd.Series)
    
    def test_bt_metrics_drawdown_in_pips(self, mt):
        expected = Decimal(-664.40).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.drawdown().min()).quantize(Decimal(DEC_PREC)) == expected
        
    def test_bt_metrics_drawdown_in_money(self, mt):
        expected = Decimal(-66.40).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.drawdown(pips_mode=False).min()).quantize(Decimal(DEC_PREC)) \
                == expected
        
    def test_bt_stagnation_period_returns_a_list_of_timedeltas(self, mt):
        stag = mt.stagnation_periods()
        assert isinstance(stag, list)
        for data in stag:
            assert isinstance(data, timedelta)
    
    def test_bt_max_stagnation_period_in_pips(self, mt):
        expected = timedelta(days=409, hours=4)
        assert max(mt.stagnation_periods()) == expected
        
    def test_bt_stagnation_period_in_money(self, mt):
        expected = timedelta(days=3743, hours=16)
        assert max(mt.stagnation_periods(pips_mode=False)) == expected
    
    @pytest.mark.xfail(__version__ < '0.2.0', reason='Bug will be corrected in version 0.2.0') 
    def test_bt_stagnation_period_in_pips_and_mone_are_consistent(self, mt):
        assert  max(mt.stagnation_periods(pips_mode=True)) == \
                max(mt.stagnation_periods(pips_mode=False))
        
    def test_bt_metrics_dd2_returns_a_series(self, mt):
        assert isinstance(mt.dd2(), pd.Series)
    
    def test_bt_metrics_dd2_in_pips(self, mt):
        expected = Decimal(-627.00).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.dd2().min()).quantize(Decimal(DEC_PREC)) == expected
        
    def test_bt_metrics_dd2_in_money(self, mt):
        expected = Decimal(-62.70).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.dd2(pips_mode=False).min()).quantize(Decimal(DEC_PREC)) == expected
        
    def test_bt_metrics_esp_returns_a_Decimal(self, mt):
        assert isinstance(mt.esp(pips_mode=True), Decimal)
        assert isinstance(mt.esp(pips_mode=False), Decimal)        
    
    def test_bt_metrics_esp_in_pips(self, mt):
        expected = Decimal(11.42).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.esp()).quantize(Decimal(DEC_PREC)) == expected
        
    def test_bt_metrics_esp_in_money(self, mt):
        expected = Decimal(1.14).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.esp(pips_mode=False)).quantize(Decimal(DEC_PREC)) == expected
    
    @pytest.mark.exposures    
    def test_bt_metrics_exposures_returns_a_tuple_of_list(self, mt):
        resp = mt.exposures()
        exp, vols = resp
        #breakpoint()
        assert isinstance(resp, tuple)
        assert isinstance(exp, list)
        assert isinstance(vols, list)
        for e in exp:
            assert isinstance(e, int)   
        for vol in vols:
            assert isinstance(vol, float) 
    
    @pytest.mark.exposures
    def test_bt_metrics_exposures_correct_value(self, mt):
        exp_expected = 7
        vols_expected = Decimal(0.07).quantize(Decimal(DEC_PREC))
        exp_act, vols_act = mt.exposures()
        assert max(exp_act) == exp_expected
        assert Decimal(max(vols_act)).quantize(Decimal(DEC_PREC)) == vols_expected        
    