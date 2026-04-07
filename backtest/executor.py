import os
import pandas as pd
import pytz
import matplotlib
from PIL import Image
from typing import Dict
from stock_data import store_to_csv, get_three_year, get_ten_year
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

from .weight_generator import Model, GpModel


# Force a headless backend so FastAPI worker threads never touch macOS GUI APIs.
os.environ.setdefault('MPLBACKEND', 'Agg')
matplotlib.use('Agg', force=True)


# This is a template for backtesting a trading strategy using QSTrader.

class Template:
    def __init__(self, AlphaModel_name: str):
        self.AlphaModel_name = AlphaModel_name
        
        # Hardcoded configuration values
        self.start_str = '2022-10-07'
        self.burn_it_str = '2023-10-07' 
        self.end_str = '2025-10-03'

        self.strategy_symbols = ['2308_3year','2317_3year','2330_3year','2382_3year','2412_3year','2454_3year','2881_3year','2882_3year','2891_3year','3711_3year']
        self.csv_dir_path = './stock_data/'
        self._ensure_data()

    def _ensure_data(self):
        """Ensure that the required stock data is available."""
        for symbol in self.strategy_symbols:
            if not os.path.exists(os.path.join(self.csv_dir_path, f"{symbol}.csv")):
                print(f"Data for {symbol} not found. Downloading...")
                stock_id = symbol.split('_')[0]
                data = get_three_year(stock_id)
                store_to_csv(stock_id, '3year', data)
                print(f"Data for {symbol} has been downloaded and stored.")
            else:
                print(f"Data for {symbol} already exists.")

    # Rest of the class implementation remains the same
    def run(self) -> Dict:
        # Duration of the backtest
        start_dt = pd.Timestamp(self.start_str, tz=pytz.UTC)
        burn_in_dt = pd.Timestamp(self.burn_it_str, tz=pytz.UTC)
        end_dt = pd.Timestamp(self.end_str, tz=pytz.UTC)

        assets = ['EQ:%s' % symbol for symbol in self.strategy_symbols]
        asset_dates = {asset: start_dt for asset in assets}
        strategy_universe = DynamicUniverse(asset_dates)

        csv_dir = os.environ.get(self.csv_dir_path[1:-2], self.csv_dir_path)

        strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=self.strategy_symbols)
        strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])

        strategy_alpha_model = Model(assets, self.csv_dir_path, self.AlphaModel_name)

        strategy_backtest = BacktestTradingSession(
            start_dt,
            end_dt,
            strategy_universe,
            strategy_alpha_model,
            rebalance='end_of_month',
            long_only=True,
            cash_buffer_percentage=0.01,
            burn_in_dt=burn_in_dt,
            data_handler=strategy_data_handler
        )
        strategy_backtest.run()

        tearsheet = TearsheetStatistics(
            strategy_equity=strategy_backtest.get_equity_curve(),
            title='alpha' + self.AlphaModel_name
        )
        result_dir = './backtest/result'
        os.makedirs(result_dir, exist_ok=True)
        result_path = f'{result_dir}/alpha{self.AlphaModel_name}.png'
        tearsheet.plot_results(filename=result_path)
        img = Image.open(result_path)
        os.remove(result_path)

        equity_df = strategy_backtest.get_equity_curve()
        res = tearsheet.get_results(equity_df)
        res = dict(sharpe=float(res['sharpe']),
                   drawdowns=float(res['drawdowns'].mean()),
                   max_drawdown=float(res['max_drawdown']),
                   max_drawdown_pct=float(res['max_drawdown_pct']),
                   max_drawdown_duration=float(res['max_drawdown_duration']),
                   equity=float(res['equity'].mean()),
                   returns=float(res['returns'].mean()),
                   cum_returns=float(res['cum_returns'].mean()),
                   image = img,
                )
        return res

class GpTemplate:
    def __init__(self, alpha_expression: str, period: str = '10y'):
        self.alpha_expression = alpha_expression
        self.period = period

        period_map = {
            '10y': ('2014-12-31', '2015-12-31', '2024-12-31'),
            '5y': ('2019-12-31', '2020-12-31', '2024-12-31'),
            '3y': ('2021-12-31', '2022-12-31', '2024-12-31'),
        }
        if period not in period_map:
            raise ValueError(f"Unknown backtest period: {period}. Use one of {list(period_map.keys())}")

        self.start_str, self.burn_it_str, self.end_str = period_map[period]
        
        self.strategy_symbols = ['0050_10year']
        self.csv_dir_path = './stock_data/'
        self._ensure_data()

    def _ensure_data(self):
        """Ensure GP backtest data exists; download if missing."""
        for symbol in self.strategy_symbols:
            csv_path = os.path.join(self.csv_dir_path, f"{symbol}.csv")
            if os.path.exists(csv_path):
                continue

            stock_id = symbol.split('_')[0]
            data = get_ten_year(stock_id)
            store_to_csv(stock_id, '10year', data)
    
    def run(self) -> Dict:
        start_dt = pd.Timestamp(self.start_str, tz=pytz.UTC)
        burn_in_dt = pd.Timestamp(self.burn_it_str, tz=pytz.UTC)
        end_dt = pd.Timestamp(self.end_str, tz=pytz.UTC)

        assets = ['EQ:%s' % symbol for symbol in self.strategy_symbols]
        asset_dates = {asset: start_dt for asset in assets}
        strategy_universe = DynamicUniverse(asset_dates)

        csv_dir = os.environ.get(self.csv_dir_path[1:-2], self.csv_dir_path)

        strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=self.strategy_symbols)
        strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])

        strategy_alpha_model = GpModel(assets, self.csv_dir_path, self.alpha_expression)

        strategy_backtest = BacktestTradingSession(
            start_dt,
            end_dt,
            strategy_universe,
            strategy_alpha_model,
            rebalance='end_of_month',
            long_only=True,
            cash_buffer_percentage=0.01,
            burn_in_dt=burn_in_dt,
            data_handler=strategy_data_handler
        )
        strategy_backtest.run()

        tearsheet = TearsheetStatistics(
            strategy_equity=strategy_backtest.get_equity_curve(),
            title=self.alpha_expression
        )

        equity_df = strategy_backtest.get_equity_curve()
        res = tearsheet.get_results(equity_df)
        res = dict(sharpe=float(res['sharpe']),
                   drawdowns=float(res['drawdowns'].mean()),
                   max_drawdown=float(res['max_drawdown']),
                   max_drawdown_pct=float(res['max_drawdown_pct']),
                   max_drawdown_duration=float(res['max_drawdown_duration']),
                   equity=float(res['equity'].mean()),
                   returns=float(res['returns'].mean()),
               cum_returns=float(res['cum_returns'].mean()),
               period=self.period,
                )
        return res
