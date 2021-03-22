import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA

# 湖北省GDP预测
time_series = pd.Series([151.0, 188.46, 199.38, 219.75, 241.55, 262.58, 328.22, 396.26, 442.04, 517.77, 626.52, 717.08,
                         824.38, 913.38, 1088.39, 1325.83, 1700.92, 2109.38, 2499.77, 2856.47, 3114.02, 3229.29,
                         3545.39, 3880.53, 4212.82, 4757.45, 5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1,
                         15967.61])
time_series.index = pd.Index(sm.tsa.datetools.dates_from_range('1978', '2010'))
time_series.plot(figsize=(12, 8))
plt.show()

# 转化为线性趋势
time_series = np.log(time_series)
time_series.plot(figsize=(8, 6))
plt.show()


# adf检验
t = sm.tsa.stattools.adfuller(time_series, )
output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                             "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)


# https://www.jianshu.com/p/4130bac8ebec