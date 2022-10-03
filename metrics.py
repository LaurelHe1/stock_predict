"""
Mean squared error for performance metrics evaluation
"""
import numpy as np
def mse_by_day(real, pred, mod):
    # open 7 hrs/d = 7 * 3600 / 5 min/d
    if mod == '5sec':
        day = 5040
    # open 7 hrs/d = 7 * 60 min/d
    elif mod == '1min':
        day = 420
    # stock open 7 hrs/d
    elif mod == '1hr':
        day = 7
    real = real.copy().reshape(-1, day)
    pred = pred.copy().reshape(-1, day)
    mse_by_day = np.nanmean((pred - real)**2, axis=1)
    return mse_by_day