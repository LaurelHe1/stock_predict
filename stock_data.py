from math import sqrt
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class StockData():
    """
    Data file: symbol (A-J ticker), open, high, low, close, average, time, day
    """
    def __init__(self, data_path: str):
        self._init_df(data_path)
        self.attribute_list = ['open', 'close', 'low', 'high', 'average']

    def _init_df(self, data_path: str):
        df = pd.read_csv(data_path)
        self.symbol_list = np.sort(df.symbol.unique())
        self.time_list = df.time.unique()
        self.day_list = df.day.unique()
        # dataframe indexed by MultiIndex after getting the time and days
        df = df.set_index(['day', 'time'])
        self.all = df
        self.open = self._get_filled_df('open')
        self.close = self._get_filled_df('close')
        self.low = self._get_filled_df('low')
        self.high = self._get_filled_df('high')
        self.average = self._get_filled_df('average')

    def _get_index(self):
        """
        Create MultiIndex from the cartesian product of day and time.
        """
        return pd.MultiIndex.from_product(
            [self.day_list, self.time_list],
            names=["day", "time"])

    def _get_filled_df(self, column_name: str):
        """
        Front fill or back fill missing data.
        """
        new_df = pd.DataFrame(0,
                              index=self._get_index(),
                              columns=self.symbol_list)
        for sym in self.symbol_list:
            new_df[sym] = self.all[self.all['symbol'] == sym][column_name]
        new_df = new_df.fillna(method='ffill')
        new_df = new_df.fillna(method='bfill')
        return new_df

    def get_slice(self, attr_name: str, day_slice: Tuple):
        """
        Get attributes if specified start and end day, as sub dataframe.
        """
        return getattr(self, attr_name).loc[day_slice[0]: day_slice[1], :]

    def plot_data(self, attr_name: str, symbol_list: List[str]):
        """
        Plot stock price vs. time.
        """
        plt.style.use("seaborn")
        for sym in symbol_list:
            y = getattr(self, attr_name)[sym].to_list()
            plt.plot(np.arange(len(y)), y, label=sym)
        plt.legend()

    def plot_corr(self, attr_name):
        """
        Calculate correlation matrix for each stock.
        """
        corrMatrix = getattr(self, attr_name)[self.symbol_list].corr()
        ax = sns.heatmap(corrMatrix, annot=True, cmap='PiYG')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.savefig(f'{attr_name}_correlation.png')
        