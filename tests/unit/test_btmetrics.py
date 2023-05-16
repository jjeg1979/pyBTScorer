"""Tests the concreate implementation of BtParser for Genbox-type Backtests"""

# Standard library imports
from pathlib import Path
from six import string_types
from decimal import Decimal
from datetime import datetime, timedelta

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
    INF,
    DEFAULT_CRITERIA,
    BtMetrics,
)


@pytest.fixture(autouse=True)
def mt():
    bt = BtGenbox(Path(r'src/payload'), BT_FILES[0])
    return BtMetrics(bt, calc_metrics_at_init=False, pips_mode=True)


# Dummy test - Courtesy of JJ
def test_btmetrics_import_works_properly():
    pass


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
        
    def test_btmetrics_all_metrics_calculated_at_init_or_not(self, mt):
        print(mt.calc_metrics_at_init)
        if not mt.calc_metrics_at_init:
            assert mt.all_metrics is not None
        else:
            assert mt.all_metrics is None
            
@pytest.mark.addedmetricsfeatures
class TestBtmetricsAddMetricsFeatures:
    """Test additional features like one metric calculation."""
    
    def test_btmetrics_calculate_one_metric_raises_index_error(self, mt):
        """_calculate_one_metric raises IndexError with a bad metric name"""
        with pytest.raises(IndexError) as er:
            mt._calculate_one_metric('non-existent-metrics-name')
        assert er.type is IndexError
        
    def test_btmetrics_max_dd_in_pips_returns_a_Decimal(self, mt):
        """max_dd should return a Decimal value."""
        assert isinstance(mt.max_dd(pips_mode=True, f='dd'), Decimal)
        
    def test_btmetrics_max_dd_in_money_returns_a_Decimal(self, mt):
        """max_dd should return a Decimal value."""
        assert isinstance(mt.max_dd(pips_mode=False, f='dd'), Decimal)
        
    def test_btmetrics_max_dd2_in_pips_returns_a_Decimal(self, mt):
        """max_dd should return a Decimal value."""
        assert isinstance(mt.max_dd(pips_mode=True, f='dd2'), Decimal)
        
    def test_btmetrics_max_dd2_in_money_returns_a_Decimal(self, mt):
        """max_dd should return a Decimal value."""
        assert isinstance(mt.max_dd(pips_mode=False, f='dd2'), Decimal)
    
    @pytest.mark.justonemetric        
    def test_btmetrics_calculate_one_metric_returns_correct_value(self, mt):
        all_metrics = mt.all_metrics
        for metric in ALL_METRICS:
            assert mt._calculate_one_metric(metric) == all_metrics[metric]
            
    def test_btmetrics_calculate_rf_in_pips_returns_a_Decimal(self, mt):
        assert isinstance(mt.calculate_rf(pips_mode=True), Decimal)
        
    def test_btmetrics_calculate_rf_in_money_returns_a_Decimal(self, mt):
        assert isinstance(mt.calculate_rf(pips_mode=False), Decimal)
        
    def test_btmetrics_calculate_rf_in_pips_returns_correct_value(self, mt):
        assert mt.calculate_rf(pips_mode=True) == Decimal(12.74).quantize(Decimal(DEC_PREC))
        
    def test_btmetrics_calculate_rf_in_money_returns_correct_value(self, mt):
        assert mt.calculate_rf(pips_mode=False) == Decimal(12.74).quantize(Decimal(DEC_PREC))
        
    def test_btmetrics_calculate_rf_in_pips_and_in_money_are_consistent(self, mt):
        assert mt.calculate_rf(pips_mode=True) == mt.calculate_rf(pips_mode=False)
    
    def test_btmetrics_num_ops_returns_an_int(self, mt):
        assert isinstance(mt.num_ops, int)        
    
    def test_btmetrics_num_ops_returns_correct_value(self, mt):
        assert mt.num_ops == 338    
            
    def test_btmetrics_is_valid_correctly_classifies_backtest(self, mt):        
        assert mt.is_valid(DEFAULT_CRITERIA) == False
        

@pytest.mark.metricsvalues
class TestBtMetricsCalculations:
    """Test the correct calculations from the BTMetrics class"""

    def test_btmetrics_calculate_pf_returns_a_decimal(self, mt):
        assert isinstance(mt.calculate_pf(), Decimal)

    def test_btmetrics_calculate_pf_in_pips(self, mt):
        expected = Decimal(1.84).quantize(Decimal(DEC_PREC))
        assert mt.calculate_pf() == expected

    def test_btmetrics_calculate_pf_in_money(self, mt):
        expected = Decimal(1.84).quantize(Decimal(DEC_PREC))
        assert mt.calculate_pf(pips_mode=False) == expected

    def test_btmetrics_drawdown_returns_a_series(self, mt):
        assert isinstance(mt.drawdown(), pd.Series)

    def test_btmetrics_drawdown_in_pips(self, mt):
        expected = Decimal(-664.40).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.drawdown().min()).quantize(Decimal(DEC_PREC)) == expected

    def test_btmetrics_drawdown_in_money(self, mt):
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
        assert max(mt.stagnation_periods(pips_mode=True)) == \
               max(mt.stagnation_periods(pips_mode=False))

    def test_btmetrics_dd2_returns_a_series(self, mt):
        assert isinstance(mt.dd2(), pd.Series)

    def test_btmetrics_dd2_in_pips(self, mt):
        expected = Decimal(-627.00).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.dd2().min()).quantize(Decimal(DEC_PREC)) == expected

    def test_btmetrics_dd2_in_money(self, mt):
        expected = Decimal(-62.70).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.dd2(pips_mode=False).min()).quantize(Decimal(DEC_PREC)) == expected

    def test_btmetrics_esp_returns_a_Decimal(self, mt):
        assert isinstance(mt.esp(pips_mode=True), Decimal)
        assert isinstance(mt.esp(pips_mode=False), Decimal)

    def test_btmetrics_esp_in_pips(self, mt):
        expected = Decimal(11.42).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.esp()).quantize(Decimal(DEC_PREC)) == expected

    def test_btmetrics_esp_in_money(self, mt):
        expected = Decimal(1.14).quantize(Decimal(DEC_PREC))
        assert Decimal(mt.esp(pips_mode=False)).quantize(Decimal(DEC_PREC)) == expected

    @pytest.mark.exposures
    def test_btmetrics_exposures_returns_a_tuple_of_list(self, mt):
        resp = mt.exposures()
        exp, vols = resp        
        assert isinstance(resp, tuple)
        assert isinstance(exp, list)
        assert isinstance(vols, list)
        for e in exp:
            assert isinstance(e, int)
        for vol in vols:
            assert isinstance(vol, float)

    @pytest.mark.exposures
    def test_btmetrics_exposures_correct_value(self, mt):
        exp_expected = 7
        vols_expected = Decimal(0.07).quantize(Decimal(DEC_PREC))
        exp_act, vols_act = mt.exposures()
        assert max(exp_act) == exp_expected
        assert Decimal(max(vols_act)).quantize(Decimal(DEC_PREC)) == vols_expected

    @pytest.mark.strikes
    def test_btmetrics_max_losing_strikes_in_pips_returns_int(self, mt):
        assert isinstance(mt.get_max_losing_strike(pips_mode=True), int)

    @pytest.mark.strikes
    def test_btmetrics_max_losing_strikes_in_money_returns_int(self, mt):
        assert isinstance(mt.get_max_losing_strike(pips_mode=False), int)

    @pytest.mark.strikes
    def test_btmetrics_max_losing_strikes_in_pips_correct_value(self, mt):
        assert mt.get_max_losing_strike(pips_mode=True) == 9

    @pytest.mark.strikes
    def test_btmetrics_max_losing_strikes_in_money_correct_value(self, mt):
        assert mt.get_max_losing_strike(pips_mode=False) == 9

    @pytest.mark.strikes
    def test_btmetrics_max_losing_strikes_in_pips_and_money_are_consistent(self, mt):
        assert mt.get_max_losing_strike(pips_mode=True) == \
            mt.get_max_losing_strike(pips_mode=False)

    @pytest.mark.strikes
    def test_btmetrics_max_winning_strikes_in_pips_returns_int(self, mt):
        assert isinstance(mt.get_max_winning_strike(pips_mode=True), int)

    @pytest.mark.strikes
    def test_btmetrics_max_winning_strikes_in_money_returns_int(self, mt):
        assert isinstance(mt.get_max_winning_strike(pips_mode=False), int)

    @pytest.mark.strikes
    def test_btmetrics_max_losing_winning_in_pips_correct_value(self, mt):
        assert mt.get_max_winning_strike(pips_mode=True) == 12

    @pytest.mark.strikes
    def test_btmetrics_max_winning_strikes_in_money_correct_value(self, mt):
        assert mt.get_max_winning_strike(pips_mode=False) == 12

    @pytest.mark.strikes
    def test_btmetrics_max_winning_strikes_in_pips_and_money_are_consistent(self, mt):
        assert mt.get_max_winning_strike(pips_mode=True) == \
               mt.get_max_winning_strike(pips_mode=False)

    @pytest.mark.strikes
    def test_btmetrics_avg_losing_strikes_in_pips_returns_Decimal(self, mt):
        assert isinstance(mt.get_avg_losing_strike(pips_mode=True), Decimal)

    @pytest.mark.strikes
    def test_btmetrics_avg_losing_strikes_in_money_returns_Decimal(self, mt):
        assert isinstance(mt.get_avg_losing_strike(pips_mode=False), Decimal)

    @pytest.mark.strikes
    def test_btmetrics_avg_losing_strikes_in_pips_correct_value(self, mt):
        assert mt.get_avg_losing_strike(pips_mode=True) == Decimal(3.43).quantize(Decimal(DEC_PREC))

    @pytest.mark.strikes
    def test_btmetrics_avg_losing_strikes_in_money_correct_value(self, mt):
        assert mt.get_avg_losing_strike(pips_mode=False) == Decimal(3.31).quantize(Decimal(DEC_PREC))

    @pytest.mark.strikes
    def test_btmetrics_avg_winning_strikes_in_pips_returns_Decimal(self, mt):
        assert isinstance(mt.get_avg_winning_strike(pips_mode=True), Decimal)

    @pytest.mark.strikes
    def test_btmetrics_avg_winning_strikes_in_money_returns_Decimal(self, mt):
        assert isinstance(mt.get_avg_winning_strike(pips_mode=False), Decimal)

    @pytest.mark.strikes
    def test_btmetrics_avg_winning_strikes_in_pips_correct_value(self, mt):
        assert mt.get_avg_winning_strike(pips_mode=True) == Decimal(2.74).quantize(Decimal(DEC_PREC))

    @pytest.mark.strikes
    def test_btmetrics_avg_winning_strikes_in_money_correct_value(self, mt):
        assert mt.get_avg_winning_strike(pips_mode=False) == Decimal(2.70).quantize(Decimal(DEC_PREC))

    @pytest.mark.lots
    def test_btmetrics_max_lots_returns_Decimal(self, mt):
        assert isinstance(mt.get_max_lots(), Decimal)

    @pytest.mark.lots
    def test_btmetrics_max_lots_returns_correct_value(self, mt):
        assert mt.get_max_lots() == Decimal(0.01).quantize(Decimal(DEC_PREC))

    @pytest.mark.lots
    def test_btmetrics_min_lots_returns_Decimal(self, mt):
        assert isinstance(mt.get_min_lots(), Decimal)

    @pytest.mark.lots
    def test_btmetrics_min_lots_returns_correct_value(self, mt):
        assert mt.get_min_lots() == Decimal(0.01).quantize(Decimal(DEC_PREC))

    @pytest.mark.lots
    def test_btmetrics_max_lots_greater_or_equal_than_min_lots(self, mt):
        assert mt.get_max_lots() >= mt.get_min_lots()

    @pytest.mark.timerelated
    def test_btmetrics_time_in_market_returns_tuple_of_ints(self, mt):
        time_in_market = mt.calculate_time_in_market()
        assert isinstance(time_in_market, tuple)
        for item in time_in_market:
            assert isinstance(item, int)

    @pytest.mark.timerelated
    def test_btmetrics_time_in_market_returns_correct_values(self, mt):
        days, hours, minutes, seconds = mt.calculate_time_in_market()
        assert days == 246
        assert hours == 20
        assert minutes == 0
        assert seconds == 0

    @pytest.mark.pct
    def test_btmetrics_pct_win_in_pips_returns_Decimal(self, mt):
        assert isinstance(mt.pct_win(pips_mode=True), Decimal)

    @pytest.mark.pct
    def test_btmetrics_pct_win_in_money_returns_Decimal(self, mt):
        assert isinstance(mt.pct_win(pips_mode=False), Decimal)

    @pytest.mark.pct
    def test_btmetrics_pct_win_in_pips_returns_correct_value(self, mt):
        assert mt.pct_win(pips_mode=True) == Decimal(47.63).quantize(Decimal(DEC_PREC))

    @pytest.mark.pct
    def test_btmetrics_pct_win_in_money_returns_correct_value(self, mt):
        assert mt.pct_win(pips_mode=False) == Decimal(47.04).quantize(Decimal(DEC_PREC))

    @pytest.mark.pct
    @pytest.mark.xfail(__version__ < "0.3.0", reason="Bug identified but won't be fixed until version 0.3.0")
    def test_btmetrics_pct_win_in_pips_and_in_money_are_consistent(self, mt):
        assert mt.pct_win(pips_mode=True) == mt.pct_win(pips_mode=False)

    @pytest.mark.pct
    def test_btmetrics_pct_loss_in_pips_returns_Decimal(self, mt):
        assert isinstance(mt.pct_loss(pips_mode=True), Decimal)

    @pytest.mark.pct
    def test_btmetrics_pct_loss_in_money_returns_Decimal(self, mt):
        assert isinstance(mt.pct_loss(pips_mode=False), Decimal)

    @pytest.mark.pct
    def test_btmetrics_pct_loss_in_pips_returns_correct_value(self, mt):
        assert mt.pct_loss(pips_mode=True) == Decimal(52.37).quantize(Decimal(DEC_PREC))

    @pytest.mark.pct
    def test_btmetrics_pct_loss_in_money_returns_correct_value(self, mt):
        assert mt.pct_loss(pips_mode=False) == Decimal(52.96).quantize(Decimal(DEC_PREC))

    @pytest.mark.pct
    @pytest.mark.xfail(__version__ < "0.3.0", reason="Bug identified but won't be fixed until version 0.3.0")
    def test_btmetrics_pct_loss_in_pips_and_in_money_are_consistent(self, mt):
        assert mt.pct_loss(pips_mode=True) == mt.pct_loss(pips_mode=False)

    @pytest.mark.pct
    def test_btmetrics_pct_loss_and_mt_pct_win_in_pips_add_up_to_one_hundred(self, mt):
        assert mt.pct_win(pips_mode=True) + mt.pct_loss(pips_mode=True) == Decimal(100)

    @pytest.mark.pct
    def test_btmetrics_pct_loss_and_mt_pct_win_in_money_add_up_to_one_hundred(self, mt):
        assert mt.pct_win(pips_mode=False) + mt.pct_loss(pips_mode=False) == Decimal(100)

    @pytest.mark.closingdays
    def test_btmetrics_calculate_closing_days_returns_an_int(self, mt):
        assert isinstance(mt.calculate_closing_days(), int)

    @pytest.mark.closingdays
    def test_btmetrics_calculate_closing_days_returns_correct_value(self, mt):
        assert mt.calculate_closing_days() == 200

    @pytest.mark.sqn
    def test_btmetrics_calculate_sqn_in_pips_returns_Decimal(self, mt):
        assert isinstance(mt.calculate_sqn(pips_mode=True), Decimal)

    @pytest.mark.sqn
    def test_btmetrics_calculate_sqn_in_money_returns_Decimal(self, mt):
        assert isinstance(mt.calculate_sqn(pips_mode=False), Decimal)

    @pytest.mark.sqn
    def test_btmetrics_calculate_sqn_in_pips_returns_correct_value(self, mt):
        assert mt.calculate_sqn(pips_mode=True) == Decimal(3.88).quantize(Decimal(DEC_PREC))

    @pytest.mark.sqn
    def test_btmetrics_calculate_sqn_in_money_returns_correct_value(self, mt):
        assert mt.calculate_sqn(pips_mode=False) == Decimal(3.88).quantize(Decimal(DEC_PREC))

    @pytest.mark.sqn
    def test_btmetrics_calculate_sqn_in_pips_and_in_money_are_consistent(self, mt):
        assert mt.calculate_sqn(pips_mode=True) == mt.calculate_sqn(pips_mode=False)

    @pytest.mark.sharpe
    def test_btmetrics_calculate_sharpe_in_pips_returns_Decimal(self, mt):
        assert isinstance(mt.calculate_sharpe(pips_mode=True), Decimal)

    @pytest.mark.sharpe
    def test_btmetrics_calculate_sharpe_in_money_returns_Decimal(self, mt):
        assert isinstance(mt.calculate_sharpe(pips_mode=False), Decimal)

    @pytest.mark.sharpe
    def test_btmetrics_calculate_sharpe_in_pips_returns_correct_value(self, mt):
        assert mt.calculate_sharpe(pips_mode=True) == Decimal(71.33).quantize(Decimal(DEC_PREC))

    @pytest.mark.sharpe
    def test_calculate_sharpe_in_money_returns_correct_value(self, mt):
        assert mt.calculate_sharpe(pips_mode=False) == Decimal(71.33).quantize(Decimal(DEC_PREC))

    @pytest.mark.sharpe
    def test_btmetrics_calculate_sharpe_in_pips_and_in_money_are_consistent(self, mt):
        assert mt.calculate_sharpe(pips_mode=True) == mt.calculate_sharpe(pips_mode=False)

    @pytest.mark.kratio
    def test_btmetrics_calculate_kratio_in_pips_returns_Decimal(self, mt):
        assert isinstance(mt.calculate_kratio(pips_mode=True), Decimal)

    @pytest.mark.kratio
    def test_calculate_kratio_in_money_returns_Decimal(self, mt):
        assert isinstance(mt.calculate_kratio(pips_mode=False), Decimal)

    @pytest.mark.kratio
    def test_btmetrics_calculate_kratio_in_pips_returns_correct_value(self, mt):
        assert mt.calculate_kratio(pips_mode=True) == Decimal(0.19).quantize(Decimal(DEC_PREC))

    @pytest.mark.kratio
    def test_btmetrics_calculate_kratio_in_money_returns_correct_value(self, mt):
        assert mt.calculate_kratio(pips_mode=False) == Decimal(0.19).quantize(Decimal(DEC_PREC))

    @pytest.mark.kratio
    def test_btmetrics_calculate_kratio_in_pips_and_in_money_are_consistent(self, mt):
        assert mt.calculate_kratio(pips_mode=True) == mt.calculate_kratio(pips_mode=False)

    @pytest.mark.operations
    def test_btmetrics_calculate_best_operation_in_pips_returns_a_tuple(self, mt):
        magnitude, moment = mt.best_operation(pips_mode=True)
        assert isinstance(mt.best_operation(pips_mode=True), tuple)
        assert isinstance(magnitude, Decimal)
        assert isinstance(moment, datetime)

    @pytest.mark.operations
    def test_btmetrics_calculate_best_operation_in_money_returns_a_tuple(self, mt):
        magnitude, moment = mt.best_operation(pips_mode=False)
        assert isinstance(mt.best_operation(pips_mode=False), tuple)
        assert isinstance(magnitude, Decimal)
        assert isinstance(moment, datetime)

    @pytest.mark.operations
    def test_btmetrics_calculate_best_operation_in_pips_returns_correct_values(self, mt):
        magnitude, moment = mt.best_operation(pips_mode=True)
        exp_moment = datetime(year=2020,
                              month=4,
                              day=7,
                              hour=12,
                              minute=0,
                              second=0)
        assert magnitude == Decimal(1793.0).quantize(Decimal(DEC_PREC))
        assert moment == exp_moment

    @pytest.mark.operations
    def test_btmetrics_calculate_worst_operation_in_pips_returns_a_tuple(self, mt):
        magnitude, moment = mt.worst_operation(pips_mode=True)
        assert isinstance(mt.worst_operation(pips_mode=True), tuple)
        assert isinstance(magnitude, Decimal)
        assert isinstance(moment, datetime)

    @pytest.mark.operations
    def test_btmetrics_calculate_worst_operation_in_money_returns_a_tuple(self, mt):
        magnitude, moment = mt.worst_operation(pips_mode=False)
        assert isinstance(mt.worst_operation(pips_mode=False), tuple)
        assert isinstance(magnitude, Decimal)
        assert isinstance(moment, datetime)

    @pytest.mark.operations
    def test_btmetrics_calculate_worst_operation_in_pips_returns_correct_values(self, mt):
        magnitude, moment = mt.worst_operation(pips_mode=True)
        exp_moment = datetime(year=2011,
                              month=11,
                              day=16,
                              hour=4,
                              minute=0,
                              second=0)
        assert magnitude == Decimal(-234.80).quantize(Decimal(DEC_PREC))
        assert moment == exp_moment

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_win_in_pips_returns_a_Decimal(self, mt):
        assert isinstance(mt.calculate_avg_win(pips_mode=True), Decimal)

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_win_in_money_returns_a_Decimal(self, mt):
        assert isinstance(mt.calculate_avg_win(pips_mode=False), Decimal)

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_win_in_pips_returns_correct_value(self, mt):
        assert mt.calculate_avg_win(pips_mode=True) == Decimal(52.23).quantize(Decimal(DEC_PREC))

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_win_in_money_returns_correct_value(self, mt):
        assert mt.calculate_avg_win(pips_mode=False) == Decimal(5.16).quantize(Decimal(DEC_PREC))

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_loss_in_pips_returns_a_Decimal(self, mt):
        assert isinstance(mt.calculate_avg_loss(pips_mode=True), Decimal)

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_loss_in_money_returns_a_Decimal(self, mt):
        assert isinstance(mt.calculate_avg_loss(pips_mode=False), Decimal)

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_loss_in_pips_returns_correct_value(self, mt):
        assert mt.calculate_avg_loss(pips_mode=True) == Decimal(-26.14).quantize(Decimal(DEC_PREC))

    @pytest.mark.operations
    def test_btmetrics_calculate_avg_loss_in_money_returns_correct_value(self, mt):
        assert mt.calculate_avg_loss(pips_mode=False) == Decimal(-2.64).quantize(Decimal(DEC_PREC))

    @pytest.mark.timerelated
    def test_btmetrics_calculate_total_time_returns_a_timedelta(self, mt):
        assert isinstance(mt.calculate_total_time(), timedelta)

    @pytest.mark.timerelated
    def test_btmetrics_calculate_total_time_returns_correct_value(self, mt):
        expected = timedelta(days=4322,
                             hours=20,
                             minutes=0)
        assert mt.calculate_total_time() == expected

    @pytest.mark.money
    def test_btmetrics_gross_profit_in_pips_returns_a_Decimal(self, mt):
        assert isinstance(mt.gross_profit(pips_mode=True), Decimal)

    @pytest.mark.money
    def test_btmetrics_gross_profit_in_money_returns_a_Decimal(self, mt):
        assert isinstance(mt.gross_profit(pips_mode=False), Decimal)

    @pytest.mark.money
    def test_btmetrics_gross_profit_in_pips_returns_correct_value(self, mt):
        assert mt.gross_profit(pips_mode=True) == Decimal(8462.00).quantize(Decimal(DEC_PREC))

    @pytest.mark.money
    @pytest.mark.xfail(reason='Known bug, but unknown reason with calculations in monetary terms')
    def test_btmetrics_gross_profit_in_money_returns_correct_value(self, mt):
        assert mt.gross_profit(pips_mode=True) == Decimal(846.10).quantize(Decimal(DEC_PREC))

    @pytest.mark.money
    def test_btmetrics_gross_loss_in_pips_returns_a_Decimal(self, mt):
        assert isinstance(mt.gross_loss(pips_mode=True), Decimal)

    @pytest.mark.money
    def test_btmetrics_gross_loss_in_money_returns_a_Decimal(self, mt):
        assert isinstance(mt.gross_loss(pips_mode=False), Decimal)

    @pytest.mark.money
    def test_btmetrics_gross_loss_in_pips_returns_correct_value(self, mt):
        assert mt.gross_loss(pips_mode=True) == Decimal(-4601.50).quantize(Decimal(DEC_PREC))

    @pytest.mark.money
    @pytest.mark.xfail(reason='Known bug, but unknown reason with calculations in monetary terms')
    def test_btmetrics_gross_loss_in_money_returns_correct_value(self, mt):
        assert mt.gross_loss(pips_mode=True) == Decimal(-460.00).quantize(Decimal(DEC_PREC))

@pytest.mark.exporttofile
class TestBtmetricsExportMetrics:
    """Tests the export capabilities of the metrics class"""
    pass