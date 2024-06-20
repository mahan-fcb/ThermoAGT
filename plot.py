import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import pearsonr
from sklearn import metrics

a = pd.read_csv('f350.csv')
x = a['ddg']
y = a['prediction']

def test_func(x, a, b):
    return (a * x) + b

params, params_covariance = optimize.curve_fit(test_func, x, y,
                                               p0=[0, 0])

print(params)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


fig = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    
ax_main.scatter(x,y,marker='o')
ax_main.plot(x, test_func(x, params[0], params[1]))
ax_main.set(ylabel='Direct $\Delta$GG (Kcal/mol)',xlabel='Reverse $\Delta$GG (Kcal/mol)')
ax_xDist.hist(x,bins=80,align='mid')
ax_xDist.set(ylabel='count')
ax_xCumDist = ax_xDist.twinx()

ax_yDist.hist(y,bins=80,orientation='horizontal',align='mid')
ax_yDist.set(xlabel='count')
ax_yCumDist = ax_yDist.twiny()

plt.savefig("rev-fw-sym.png",bbox_inches="tight",edgecolor='none',pad_inches=0.02,dpi=500)
