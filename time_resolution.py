"""
Compare time resoultions of original (5 seconds), 1 minute and 1 hour.
"""
from stock_data import StockData
data = StockData('train_data.csv')
attr_name = 'open'

agg_inter = 12 # 1 minute 60/5
agg_inter = 720 # 1 hour: 3600/5
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
for sym in data.symbol_list:
    train_data = np.array(data.get_slice(attr_name=attr_name,
                            day_slice=(0, 86))[sym].to_list())
    
    by_min = np.mean(train_data.reshape(-1, 12), axis=1)
    by_hour = np.mean(train_data.reshape(-1, 720), axis=1)
    fig, axs = plt.subplots(3)
    fig.suptitle(f'Comparison of different time resolution for {sym}', fontsize=16)
    axs[0].plot(train_data)
    axs[1].plot(by_min)
    axs[2].plot(by_hour)
    plt.savefig(f'time_resolution_{sym}.png')
    break
