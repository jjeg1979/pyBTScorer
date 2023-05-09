# Standard library imports
from collections import Counter
import datetime as dt
from typing import Tuple, Any, Set, List
from decimal import Decimal

# Non-standard library imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Project imports

class Btmetrics:
    """
    Represents a Btmetrics object. From a backtest, it calculates different metrics to characterize the
    backtest and to ensure that the

    Instance variables:
       ops (pandas.DataFrame): Operations extracted from backtest in a Pandas DataFrame       
    
    Instance properties:    
        * operations
        * metrics

    Instance methods:
        * selected_metrics
        * is_valid
        * calculate_pf
        * drawdown
        * stagnation_period
        * dd2
        * esp
        * exposures
        * get_strikes
        * get_max_losing_strike
        * get_max_winning_strike
        * get_avg_losing_strike
        * get_avg_winning_strike
        * get_max_lots
        * get_min_lots
        * calculate_time_in_market
        * pct_win
        * pct_loss
        * calculate_closing_days
        * calculate_sqn
        * calculate_sharpe
        * best_operation
        * worst_operation
        * calculate_avg_win
        * calculate_avg_loss
        * calculate_total_time
        * gross_profit
        * gross_loss
        * kratio
    """

    def __init__(self, ops: pd.DataFrame) -> None:
        """Creates and returns a Metrics object

        Args:
            ops (pandas.DataFrame): Operations extracted from backtest 
                                    in a Pandas DataFrame      
        
        Returns:
            None

        TODO:   Include pips_mode as argument for the constructor in order
                to homogenize the call to the metrics functions
        """
        self._ops = ops
        self._available_metrics = self.available_metrics
        self._all_metrics = self.calculate_metrics()

    
    @property
    def operations(self) -> pd.DataFrame:
        """Operations in the backtest. The data available for the operations are:
                * Open Time
                * Close Time
                * Pips
                * Profit
                * ...
            
            Args:
            
            Returns:
                (pandas.DataFrame): DataFrame with the operations
        """
        if self._ops is None:
            raise ValueError
        else:
            return self._ops
    
    @operations.setter
    def operations(self, value: pd.DataFrame) -> None:
        self._ops = value if value is pd.DataFrame else None    

    @property
    def all_metrics(self) -> dict:
        return self._calculate_metrics()   
    
    @property
    def available_metrics(self) -> Set[str]:
        """Property that returns a dict with the names of the metrics and the corresponding function
        in this class that calculates the metrics.

        Returns:
            (dict): Dictionary with the following structure:
                    {'metric_name': metric_function_name }
                    For example:
                        {'PF': self.calculate_pf}
        """
        # return {
        #     'PF'                  : self.calculate_pf,                  # Profit Factor
        #     'EP'                  : self.esp,                           # Expectancy
        #     'DD'                  : self.drawdown,                      # Drawdown
        #     'Stagnation Period'   : self.stagnation_period,             
        #     'DD2'                 : self.dd2,                           
        #     'Max. Exposure'       : self.exposures,                                             
        #     'Max. Losing Strike'  : self.get_max_losing_strike,         
        #     'Max. Winning Strike' : self.get_max_winning_strike,
        #     'Avg. Losing Strike'  : self.get_avg_losing_strike,
        #     'Avg. Winning Strike' : self.get_avg_winning_strike,
        #     'Max. Lots'           : self.get_max_lots,
        #     'Min. Lots'           : self.get_min_lots,
        #     'Time in Market'      : self.calculate_time_in_market,
        #     'Pct. Win'            : self.pct_win,
        #     'Pct. Loss'           : self.pct_loss,
        #     'Closing Days'        : self.calculate_closing_days,
        #     'SQN'                 : self.calculate_sqn,
        #     'Sharpe'              : self.calculate_sharpe,
        #     'Best Op'             : self.best_operation,
        #     'Worst Op'            : self.worst_operation,
        #     'Avg Win:'            : self.calculate_avg_win,
        #     'Avg Loss'            : self.calculate_avg_loss,
        #     'Backtest Time'       : self.calculate_total_time,             # Total time for the bactest
        #     'Gross Profit'        : self.gross_profit,
        #     'Gross Loss'          : self.gross_loss,
        #     'Kratio'              : self.kratio,
        # }
        return {key for key in self.all_metrics.keys()}
    
    def _calculate_metrics(self, pips_mode=True) -> dict:
        """Method that returns a dict with the values for the matrics included in this class

        Args:
            pips_mode (bool): To determine if metrics are expressed in pips or in money
                              Defaults to True

        Returns:
            (dict): Dictionary with the following structure:
                    {'metric_name': metric_value }
                    For example:
                        {'PF': value,
                         ...
                        }
        """
        strikes = self._get_strikes(pips_mode)
        return {
            'PF'                  : self.calculate_pf(pips_mode),                  
            'EP'                  : self.esp(pips_mode),                           
            'DD'                  : self.drawdown(pips_mode),                      
            'Stagnation Period'   : self.stagnation_period(pips_mode),             
            'DD2'                 : self.dd2(pips_mode),                           
            'Max. Exposure'       : max(self.exposures()),
            'Max. Losing Strike'  : self.get_max_losing_strike(strikes),         
            'Max. Winning Strike' : self.get_max_winning_strike(strikes),
            'Avg. Losing Strike'  : self.get_avg_losing_strike(strikes),
            'Avg. Winning Strike' : self.get_avg_winning_strike(strikes),
            'Max. Lots'           : self.get_max_lots(),
            'Min. Lots'           : self.get_min_lots(),
            'Time in Market'      : self.calculate_time_in_market(),
            'Pct. Win'            : self.pct_win(pips_mode),
            'Pct. Loss'           : self.pct_loss(pips_mode),
            'Closing Days'        : self.calculate_closing_days(),
            'SQN'                 : self.calculate_sqn(pips_mode),
            'Sharpe'              : self.calculate_sharpe(pips_mode),
            'Best Op'             : self.best_operation(pips_mode),
            'Worst Op'            : self.worst_operation(pips_mode),
            'Avg Win:'            : self.calculate_avg_win(pips_mode),
            'Avg Loss'            : self.calculate_avg_loss(pips_mode),
            'Backtest Time'       : self.calculate_total_time(),             
            'Gross Profit'        : self.gross_profit(pips_mode),
            'Gross Loss'          : self.gross_loss(pips_mode),
            'Kratio'              : self.kratio(pips_mode),
        }

    def selected_metrics(self, selected_metrics: List[str]) -> dict:
        """Method that returns a list with the values of the selected metrics.
        
        Args:
            selected_metrics List[str]: List that contains the names of the selected metrics

        Returns:
            dict:   List with the values of the selected metrics
                    {'metric_name': metric_value }
                    For example:
                        {'PF': value,
                         'EP': value,
                         'DD2': value,
                         ...
                        }
        """
        # First check if we want to retrieve all the metrics
        if len(selected_metrics) == len(self.avilable_metrics):
            return self.all_metrics
        
        return {metric:self.all_metrics[metric] for metric in selected_metrics}

    def is_valid(self, criteria: dict) -> bool:
        pass


    def calculate_pf(self, pips_mode=True) -> Decimal:
        """Calculates the Profit Factor for the backtest operations

        Args:
            pips_mode (bool): Indicates whether the results must be in Pips or in monetary terms

        Returns:
            (Decimal): Value for the Profit Factor
        """
        column = 'Pips' if pips_mode else 'Profit'
        profit = self.operations[self.operations[column] > 0][column].sum()
        loss = -self.operations[self.operations[column] < 0][column].sum()
        return profit / loss if loss > 0 else np.inf()


    def drawdown(self, pips_mode=True) -> pd.Series:
        """Calculates the Drawdown 

        Args:
            pips_mode (bool): Indicates whether the results must be in Pips or in monetary terms

        Returns:
            np.array: Numpy array with the drawdown values
        """
        self.operations = self.operations
        column = 'Pips' if pips_mode else 'Profit'
        return self.operations[column].cumsum() - self.operations[column].cumsum().cummax()


    def stagnation_period(self, mode_pips=True) -> List[float]:  
        """Calculates the periods where the balance curve is not increasing 

        Args:
            pips_mode (bool): Indicates whether the results must be in Pips or in monetary terms

        Returns:
            (list): List with the stagnation durations
        """      
        dd = self.drawdown(mode_pips).to_list()
        stagnation = [self.operations['Close Time'].iloc[dd.index(d)] for d in dd if d != 0]

        durations = pd.Series(stagnation).diff().to_list()

        # Need to remove first value (NaT)
        return durations[1:]


    def dd2(self, pips_mode=True) -> np.array:
        """Calculates the Drawdown in a different way from the self.drawdown method of this class

        Args:
            pips_mode (bool): Indicates whether the results must be in Pips or in monetary terms

        Returns:
            np.array: Numpy array with the drawdown values

        TODO: - Return value should be pd.Series, to be consisten with self.drawdown method
        """
        self.operations = self.operations['Pips'] if pips_mode is True else self.operations['Profit']
        d, dd_actual = [], 0

        for p in self.operations:
            dd_actual -= p
            if p < 0:
                dd_actual = 0
            d.append(dd_actual)

        return np.array(d)


    def esp(self, pips_mode=True) -> Decimal:
        """Calculates the Mathematica Expectancy for the backtest operations

        Args:
            pips_mode (bool): Indicates whether the results must be in Pips or in monetary terms

        Returns:
            (Decimal): Value for the Expectancy
        """
        esp = self.operations['Pips'].mean() if pips_mode is True else self.operations['Profit'].mean()
        return Decimal(esp)


    def exposures(self) -> list:
        exp = []
        for _, op in self.operations.iterrows():
            open_time = op['Open Time']
            close_time = op['Close Time']
            ops = self.operations[(self.operations['Open Time'] >= open_time) & \
                                  (self.operations['Close Time'] <= close_time)]
            exp.append(ops.shape[0])
        return exp


    def _get_strikes(self, pips_mode=True) -> dict:
        """
        This method returns a Counter object where the positive and negative
        strikes are shown.
        """
        
        column = 'Pips' if pips_mode else 'Profit'
        self.operations['Strike Type'] = np.where(self.operations[column] > 0, 1, -1)
        strikes = {1: [], -1: []}
        counter = 0
        for idx in range(1, self.operations.shape[0]):
            if self.operations['Strike Type'].iloc[idx] == self.operations['Strike Type'].iloc[idx - 1]:
                counter += 1
            else:
                if self.operations['Strike Type'].iloc[idx - 1] == 1:
                    # Changed from winning strike to losing strike
                    strikes[1].append(counter)
                elif self.operations['Strike Type'].iloc[idx - 1] == -1:
                    # Changed from losing strike to winning strike
                    strikes[-1].append(counter)
                counter = 1
        return {1: Counter(strikes[1]), -1: Counter(strikes[-1])}


    def get_max_losing_strike(self, strikes: dict) -> int:
        return max(strikes[-1].keys())


    def get_max_winning_strike(self, strikes: dict) -> int:
        return max(strikes[1].keys())


    def get_avg_losing_strike(self, strikes: dict) -> float:
        pairs = zip(strikes[-1].keys(), strikes[1].values())
        average = sum(pair[0] * pair[1] for pair in pairs)
        return average / sum(strikes[-1].values())


    def get_avg_winning_strike(self, strikes: dict) -> float:
        pairs = zip(strikes[1].keys(), strikes[1].values())
        average = sum(pair[0] * pair[1] for pair in pairs)
        return average / sum(strikes[1].values())
    

    def get_max_lots(self) -> float:
        return max(self.operations['Volume'])


    def get_min_lots(self) -> float:
        return min(self.operations['Volume'])


    def _remove_overlapping_ops(self) -> pd.DataFrame:
        """
        Returns a dataframe with overlapping operations removed.
        An operation overlaps another one if and only if the former's Open Time
        and the former's Close Time is encompassed in the latter.

        This is a previous step in order to not account for duplicated time
        when calculating time in market for the backtest
        """
        indices = set(self.operations.reset_index(drop=True).index)
        for idx in indices:
            open_time = self.operations['Open Time'].iloc[idx]
            close_time = self.operations['Close Time'].iloc[idx]
            duplicated_idx = set(self.operations[(self.operations['Open Time'] >= open_time) & \
                                                 (self.operations['Close Time'] <= close_time)].index)
            # breakpoint()
            if len(duplicated_idx) > 1:
                indices = indices.difference(duplicated_idx)

        return self.operations.iloc[list(indices)]


    def calculate_time_in_market(self) -> Tuple[Any, Any, Any, Any]:      
        self.operations_clean = self.operations
        # indices = set(self.operations_clean.reset_index(drop=True).index)
        total_time = self.operations_clean.iloc[0]['Duration']
        for idx in range(1,self.operations_clean.shape[0]):
            if self.operations_clean['Open Time'].iloc[idx] < self.operations_clean['Close Time'].iloc[idx-1]:
                added_time = self.operations_clean['Close Time'].iloc[idx] - self.operations_clean['Close Time'].iloc[idx-1]
                total_time += added_time
            else:
                total_time += self.operations_clean['Duration'].iloc[idx]

        days = total_time.days
        hours = (total_time - dt.timedelta(days=days)).seconds // 3600
        minutes = (total_time - dt.timedelta(days=days, hours=hours)).seconds//60
        seconds = (total_time-dt.timedelta(days=days, hours=hours, minutes=minutes)).seconds

        return days, hours, minutes, seconds


    def pct_win(self, pips_mode=True) -> float:
        """
        pips_mode: 
        """
        column = 'Pips' if pips_mode else 'Profit'
        return self.operations[self.operations[column] > 0].shape[0] / self.operations.shape[0] * 100
    

    def pct_loss(self, pips_mode=True) -> float:
        return 100 - self.pct_winner(self.operations, pips_mode)


    def calculate_closing_days(self) -> int:
        """
        Calculates the number of different days where an order has been closed
        """
        years = pd.DatetimeIndex(self.operations['Close Time']).year.values.tolist()
        months = pd.DatetimeIndex(self.operations['Close Time']).month.values.tolist()
        days = pd.DatetimeIndex(self.operations['Close Time']).day.values.tolist()
        dates = zip(years, months, days)
        dates = set(dates)
        return len(dates)


    def calculate_sqn(self, pips_mode=True) -> float:
        if pips_mode:
            return self.operations['Pips'].mean()/(self.operations['Pips'].std()/(self.operations.shape[0] ** 0.5))
        else:
            return self.operations['Profit'].mean()/(self.operations['Profit'].std()/(self.operations.shape[0] ** 0.5))


    def calculate_sharpe(self, pips_mode=True) -> float:
        return self.calculate_sqn(self.operations, pips_mode) * (self.operations.shape[0] ** 0.5)


    def best_operation(self, pips_mode=True) -> tuple[Any, Any]:
        if pips_mode:
            column = 'Pips'
            factor = 10
        else:
            column = 'Profit'
            factor = 1
        return self.operations[column].max() * factor, \
            self.operations.iloc[self.operations[column].idxmax()]['Close Time']


    def worst_operation(self, pips_mode=True) -> tuple[Any, Any]:
        if pips_mode:
            column = 'Pips'
            factor = 1
        else:
            column = 'Profit'
            factor = 10
            
        return self.operations[column].min() * factor, \
            self.operations.iloc[self.operations[column].idxmin()]['Close Time']


    def calculate_avg_win(self, pips_mode=True) -> int:
        column = 'Pips' if pips_mode else 'Profit'
        ganancia = self.operations[self.operations[column] >= 0][column].mean()
        return int(np.round(ganancia, 0))


    def calculate_avg_loss(self, pips_mode=True) -> int:
        column = 'Pips' if pips_mode else 'Profit'
        perdida = self.operations[self.operations[column] < 0][column].mean()
        return int(np.round(perdida, 0))


    def calculate_total_time(self) -> dt.timedelta:
        inicio = self.operations['Open Time'].iloc[0]
        fin = self.operations['Close Time'].iloc[-1]
        return (fin - inicio)


    def gross_profit(self, pips_mode=True) -> int:
        column = 'Pips' if pips_mode else 'Profit'
        return self.operations[self.operations[column] >= 0][column].sum()


    def gross_loss(self, pips_mode=True) -> int:
        column = 'Pips' if pips_mode else 'Profit'
        return self.operations[self.operations[column] < 0][column].sum()


    def _eqm(self, pips_mode=True) -> Tuple[Any, Any, Any]:
        column = 'Pips' if pips_mode else 'Profit'
        x = self.operations.reset_index().index.values.reshape(-1, 1)
        y = self.operations[column].cumsum()

        xhat = np.mean(x)
        yhat = np.mean(y)
        n = x.flatten().size
        xdev = np.sum(np.square(x - xhat))
        ydev = np.sum(np.square(y - yhat))
        xydev = np.square(np.sum((x.flatten() - xhat) * (y - yhat)))
        error = np.sqrt(np.around((ydev - xydev / xdev) / (len(x) - 2), decimals=8)) / np.sqrt(xdev)

        return x, y, error


    def kratio(self, pips_mode=True) -> float:
        x, y, error = self._eqm(self.operations, pips_mode)
        model = LinearRegression().fit(x, y)
        return model.coef_[0] / (error * len(x))
