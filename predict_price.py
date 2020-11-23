# -*-coding:utf-8-*-

# MAE: ƽ������������Ԥ��������ʵ���ݼ��Ľӽ��̶ȵĳ̶ȣ�ԽСԽ��

# MSE: ���������������ݺ�ԭʼ���ݶ�Ӧ�����������ƽ���͵ľ�ֵ��ԽСԽ��
# RMSE: ��������, MSE������
# R2: �ж�ϵ�������ͻع�ģ�͵ķ���÷֣�[0,1]���ӽ�1˵���Ա���Խ�ܽ���������ķ���仯��
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sklearn.utils as su
import sklearn.metrics as sm
from sklearn import metrics

data = np.loadtxt('./data.csv', unpack=False, dtype='U20', delimiter=',')
headers = data[0, 1:]  # ȥ��ID��

x = np.array(data[1:, 1:-1], dtype=float)  # ���뼯
y = np.array(data[1:, -1], dtype=float)  # �����

# �������ݼ�(������ģ�� �������ݼ��ǳ���Ҫ)
x, y = su.shuffle(x, y, random_state=7)

# ���ֲ��Լ���ѵ����
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=99, shuffle=False)
# print(train_x.shape, test_x.shape)
names = ['Decision Tree', 'Linear Regression', 'KNN', 'RFR', 'Gradient Boost', 'Bagging']

regressors = [
    DecisionTreeRegressor(),  # ������
    LinearRegression(),  # ���Իع�
    KNeighborsRegressor(),  # KNN
    RandomForestRegressor(max_depth=10, n_estimators=100, min_samples_split=2),  # ���ɭ�ֻع�
    GradientBoostingRegressor(n_estimators=100),  # �ݶ������ع�
    BaggingRegressor(),  # ����ѧϰ Bagging
]
r2_scores = []
MSE_list = []
RMSE_list = []
cv_10_mean = []
plt.figure(figsize=(15, 8))
# print("R2_score:")
for i, model in enumerate(regressors, 1):  # i��1��ʼ
    ax = plt.subplot(3, 2, i)
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)

    # 10�۽�����֤
    # print(names[i - 1], 'cv 10 mean:', ms.cross_val_score(model, train_x, train_y, cv=10, scoring='r2').mean())
    cv = ms.cross_val_score(model, train_x, train_y, cv=10, scoring='r2').mean()
    cv_10_mean.append([cv, names[i-1]])

    MSE = metrics.mean_squared_error(test_y, pred_test_y)
    RMSE = np.sqrt(metrics.mean_squared_error(test_y, pred_test_y))
    # print(names[i - 1], MSE)
    # print(names[i - 1], RMSE)
    MSE_list.append([MSE, names[i - 1]])
    RMSE_list.append([RMSE, names[i - 1]])

    r2_score = sm.r2_score(test_y, pred_test_y)
    # print(names[i - 1], r2_score)
    r2_scores.append([r2_score, names[i - 1]])

    ax.plot(train_y, '-g', label='y_train')
    ax.plot(np.arange(train_x.shape[0], train_x.shape[0] + test_x.shape[0]), test_y, '-b', label='y_test')
    ax.plot(np.arange(train_x.shape[0], train_x.shape[0] + test_x.shape[0]), pred_test_y, '-r', label='y_predict')
    ax.text(.5, .5, '%.2f' % r2_score, fontsize=14, horizontalalignment='left', verticalalignment='top')
    ax.set_title(names[i - 1])
    ax.legend()

plt.tight_layout()
plt.show()

# 10�۽�����֤CV��ֵ����
cv_10_mean.sort()
data = list(zip(*cv_10_mean))
plt.plot(data[1], data[0], '-.o', figure=plt.figure(figsize=(15, 7)))
for a, b in cv_10_mean: plt.text(b, a, '%.2f' % a, ha='center', va='bottom', fontsize=14)
plt.title('cv_10_mean')
plt.grid()
plt.show()

# MSE�÷�����
MSE_list.sort()
data = list(zip(*MSE_list))
plt.plot(data[1], data[0], '-.o', figure=plt.figure(figsize=(15, 7)))
for a, b in MSE_list: plt.text(b, a, '%.2f' % a, ha='center', va='bottom', fontsize=14)
plt.title('MSE')
plt.grid()
plt.show()

# RMSE�÷�����
RMSE_list.sort()
data = list(zip(*RMSE_list))
plt.plot(data[1], data[0], '-.o', figure=plt.figure(figsize=(15, 7)))
for a, b in RMSE_list: plt.text(b, a, '%.2f' % a, ha='center', va='bottom', fontsize=14)
plt.title('RMSE')
plt.grid()
plt.show()

# R2�÷�����
r2_scores.sort()
data = list(zip(*r2_scores))
plt.plot(data[1], data[0], '-.o', figure=plt.figure(figsize=(15, 7)))
for a, b in r2_scores: plt.text(b, a, '%.2f' % a, ha='center', va='bottom', fontsize=14)
plt.title('R2_Score')
plt.grid()
plt.show()

print("OK!")