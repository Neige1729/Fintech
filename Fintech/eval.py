import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp

# 假设你有以下数据：
# y_true 表示真实标签
# y_pred 表示模型预测的标签
from pandas import Series, DataFrame

df = pd.read_csv("true.csv")
print(df)
y_true = df['real'].values
print(y_true)
y_pred = df['pred'].values
print(y_pred)
y_0 = df['0'].values
print(y_0)
y_1 = df['1'].values
print(y_1)


def PlotLift(preds, labels, n, asc):
    # preds is score: asc=1
    # preds is prob: asc=0

    pred = preds  # 预测值
    bad = labels  # 取1为bad, 0为good
    lftds = DataFrame({'bad': bad, 'pred': pred})
    lftds['good'] = 1 - lftds.bad

    if asc == 1:
        lftds1 = lftds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        lftds1 = lftds.sort_values(by=['pred', 'bad'], ascending=[False, True])

    lftds1.index = range(len(lftds1.pred))

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    lft_index = Series(lftds1.index)
    lft_index = lft_index.quantile(q=qe)
    lft_index = np.ceil(lft_index).astype(int)
    lft_index = list(lft_index)

    Rate_all = 1.0 * sum(lftds1.bad) / len(lftds1.bad)

    lift = []
    temp = lftds1.loc[lftds1.index <= lft_index[0]]
    lift = lift + [1.0 * sum(temp.bad) / len(temp.bad) / Rate_all]

    for i in range(1, len(lft_index)):
        temp = lftds1.loc[(lftds1.index <= lft_index[i]) & (lftds1.index > lft_index[i - 1])]
        lift = lift + [1.0 * sum(temp.bad) / len(temp.bad) / Rate_all]

    lift_no = range(1, len(lift) + 1)

    plt.plot(lift_no, lift, 'bo-', linewidth=2)
    plt.axhline(1, color='gray', linestyle='--')
    plt.title('Lift-chart', fontsize=15)
    plt.show()

PlotLift(y_0, y_true, 30, 1)

def calculate_lift(predictions, actual):
    # 计算模型的预测率
    model_rate = np.mean(predictions)
    # 计算实际的预测率
    actual_rate = np.mean(actual)
    # 计算 LIFT 值
    lift = model_rate / actual_rate
    return lift

def calculate_coverage(predictions, actual):
    # 计算预测结果的覆盖率
    coverage = np.mean(predictions == actual)
    return coverage

# 计算模型的 LIFT 值
lift = calculate_lift(y_pred, y_true)
print(lift)

# 计算模型的覆盖率
coverage = calculate_coverage(y_pred, y_true)
print(coverage)

def ks_calc_cross(data, pred, y_label):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'crossdens': 好坏客户累积概率分布以及其差值gap
    '''
    crossfreq = pd.crosstab(data[pred[0]], data[y_label[0]])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
    return ks, crossdens


def ks_calc_auc(data, pred, y_label):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值
    '''
    fpr, tpr, thresholds = roc_curve(data[y_label[0]], data[pred[0]])
    ks = max(tpr - fpr)
    return ks


def ks_calc_2samp(data, pred, y_label):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'cdf_df': 好坏客户累积概率分布以及其差值gap
    '''
    Bad = data.loc[data[y_label[0]] == 1, pred[0]]
    Good = data.loc[data[y_label[0]] == 0, pred[0]]
    data1 = Bad.values
    data2 = Good.values
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = (np.searchsorted(data2, data_all, side='right')) / (1.0 * n2)
    ks = np.max(np.absolute(cdf1 - cdf2))
    cdf1_df = pd.DataFrame(cdf1)
    cdf2_df = pd.DataFrame(cdf2)
    cdf_df = pd.concat([cdf1_df, cdf2_df], axis=1)
    cdf_df.columns = ['cdf_Bad', 'cdf_Good']
    cdf_df['gap'] = cdf_df['cdf_Bad'] - cdf_df['cdf_Good']
    return ks, cdf_df


data = {'y_label': y_true,
        'pred': y_0}

data = pd.DataFrame(data)
ks1, crossdens = ks_calc_cross(data, ['pred'], ['y_label'])

ks2 = ks_calc_auc(data, ['pred'], ['y_label'])

ks3 = ks_calc_2samp(data, ['pred'], ['y_label'])

get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
ks4 = get_ks(data['pred'], data['y_label'])
print('KS1:', ks1['gap'].values)
print('KS2:', ks2)
print('KS3:', ks3[0])
print('KS4:', ks4)