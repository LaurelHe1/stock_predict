import numpy as np
import pandas as pd
def interp_length(p1, p2, length):
    # combine period 1 and period 2
    dp = (p2 - p1) / length
    res = np.zeros(length)
    for i in range(length):
        res[i] = p1 + i * dp
    return res

def expand_pred(pred, aggr_int):
    # expand prediction array to correct 5 sec interval
    res = np.zeros(len(pred) * aggr_int)
    for i in range(1, len(pred) + 1):
        if i == len(pred):
            res[(i-1)*aggr_int: i*aggr_int] = \
                interp_length(pred[i-1], pred[i-1], aggr_int)
        else:
            res[(i-1)*aggr_int: i*aggr_int] = \
                interp_length(pred[i-1], pred[i], aggr_int)
    return res