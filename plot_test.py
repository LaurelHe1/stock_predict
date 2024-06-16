import numpy as np
import matplotlib.pyplot as plt
from stock_data import StockData
from metrics import mse_by_day

plt.style.use("seaborn")
data = StockData("train_data.csv")

# Plot test data performance for trainining data
from merge_data import expand_pred
from metrics import mse_by_day
import seaborn as sns
methods = ['baseline', 'arima', 'arima_ind', 'rf', 'lstm'] # shared model and all individual models
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
p1_mse = np.zeros((len(methods), len(data.symbol_list)))
p2_mse = np.zeros((len(methods), len(data.symbol_list)))
avg_mse = np.zeros((len(methods), len(data.symbol_list)))

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
attr_name = 'open'
fig, axs = plt.subplots(2, 5, figsize=(70, 30))
idx = 0
for sym in data.symbol_list:
    real = np.array(data.get_slice(attr_name=attr_name,
                    day_slice=(78, 86))[sym].to_list())
    row = idx // 5
    col = idx % 5
    axs[row, col].plot(real, label="real stock price")
    m_idx = 0
    for method in methods:
        pred = np.load(f'{method}_{sym}_pred.npy')
        pred = expand_pred(pred, 720) # 1 hr
        mses = mse_by_day(real=real, pred=pred, mod='5sec')
        axs[row, col].plot(pred, label=f'{method} pred price p1_mse={np.mean(mses[0:4]): .3f} p2_mse={np.mean(mses[4:]): .3f}')
        p1_mse[m_idx, idx] = np.mean(mses[0:4]) # split 9 day prediction into 2 periods, period 1: 0-3, 4 days
        p2_mse[m_idx, idx] = np.mean(mses[4:]) # period 2: 4-8, 5 days
        avg_mse[m_idx, idx] = np.mean(mses)
        m_idx += 1
    axs[row, col].set_title(f'{sym} method comparison', fontsize=36)
    axs[row, col].legend()
    idx += 1
np.savetxt("p1_mse_pred.csv", p1_mse, delimiter=",")
np.savetxt("p2_mse_pred.csv", p2_mse, delimiter=",")
np.savetxt("avg_mse_pred.csv", avg_mse, delimiter=",")

for ax in axs.flat:
    ax.set(xlabel='seconds', ylabel='price')
plt.tight_layout()
plt.savefig(f'train_data_test.png')
plt.close('all')

# Plot test data performance for actual prediction
from test_solutions import Evaluator
eval = Evaluator('test_solutions.csv')
from merge_data import expand_pred
from metrics import mse_by_day
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})

p1_mse = np.zeros((len(methods), len(data.symbol_list)))
p2_mse = np.zeros((len(methods), len(data.symbol_list)))
avg_mse = np.zeros((len(methods), len(data.symbol_list)))
attr_name = 'open'
fig, axs = plt.subplots(2, 5, figsize=(70, 30))
idx = 0
for sym in data.symbol_list:
    ind = eval.solutions[eval.solutions.index.str.contains(sym)]
    real = np.array(ind[ind.day == 0].open.to_list())
    for i in range(1, 10):
        real = np.append(real, np.array(ind[ind.day == i].open.to_list()))
    print(real.shape)
    row = idx // 5
    col = idx % 5
    axs[row, col].plot(real, label="real stock price")
    m_idx = 0
    for method in methods:
        pred = np.load(f'{method}_{sym}_pred_real.npy')
        pred = expand_pred(pred, 720) # 1 hr
        print(pred.shape)
        mses = mse_by_day(real=real, pred=pred, mod='5sec')
        axs[row, col].plot(pred, label=f'{method} pred price p1_mse={np.mean(mses[0:4]): .3f} p2_mse={np.mean(mses[4:]): .3f}')
        p1_mse[m_idx, idx] = np.mean(mses[0:4])
        p2_mse[m_idx, idx] = np.mean(mses[4:])
        avg_mse[m_idx, idx] = np.mean(mses)
        m_idx += 1
    axs[row, col].set_title(f'{sym} method comparison', fontsize=36)
    axs[row, col].legend()
    idx += 1
for ax in axs.flat:
    ax.set(xlabel='seconds', ylabel='price')

np.savetxt("p1_mse_pred_real.csv", p1_mse, delimiter=",")
np.savetxt("p2_mse_pred_real.csv", p2_mse, delimiter=",")
np.savetxt("avg_mse_pred_real.csv", avg_mse, delimiter=",")
plt.tight_layout()
plt.savefig(f'real_data_test.png')
plt.close('all')

#%%
eval.solutions.open.to_list()
# %%
import pandas as pd
submission = pd.read_csv('pred.csv', index_col="id")
submission.rename(columns={'open': 'predicted_open'}, inplace=True)
# %%
submission
# %%
joint_data = pd.concat([eval.solutions, submission], axis=1)

# %%
joint_data.open.to_list()
# %%
joint_data.predicted_open

#%%

no_j = joint_data[joint_data.index.str.contains("A")]

#%%
import matplotlib.pyplot as plt
length = len(no_j.open.to_list())
plt.plot(no_j.open.to_list(), 'b')
plt.plot(no_j.predicted_open.to_list(), 'r')
plt.show()

#%%
import numpy as np
no_j.loc[no_j.predicted_open.isnull(), "predicted_open"] = 0.
no_j["error"] = (no_j.predicted_open - no_j.open)**2
# n.b. I didn't divide by 10 in the eval equation in the instructions.
daily_avg_error = 10*no_j.groupby(["period", "day"]).error.apply(
    np.nanmean
)
period_errors = daily_avg_error.groupby("period").mean()

#%%
period_errors
# return period_errors.to_dict()
# %%
import matplotlib.pyplot as plt
length = len(joint_data.open.to_list())
plt.plot(joint_data.open.to_list()[: length - 45360], 'b')
plt.plot(joint_data.predicted_open.to_list()[: length - 45360], 'r')
plt.show()
# %%
no_j_open = joint_data.open.to_list()[: length - 45360]
no_j_pred = joint_data.predicted_open.to_list()[: length - 45360]
