# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:40:49 2024

@author: zero

"""
#%%
from tsai.all import *
import sklearn.metrics as skm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

#%%
df = pd.read_csv(r"D:\a_zyc_file\a-zouyuechao\graduation_thesis\experiment\time_series_damage_detection\dataset\MVS_P2_NM_Z\train_valid.csv",header=None)   
# df = pd.read_csv(r"D:\a_zyc_file\a-zouyuechao\graduation_thesis\experiment\time_series_damage_detection\dataset\sweden\dataset-train.csv",header=None)
#%%
# 以下得到的y没有作用，将被覆盖
X, y = df2xy(df, data_cols=None)

X=X.reshape(1280,512,40)
# X=X.reshape(5120,128,40)
# X=X.reshape(528,1000,21)
# X=X.reshape(1056,500,21)

X=np.transpose(X,(0,2,1))
#%% 512, 4608, 512
label = [0] * 64
for i in range(64, 512, 64):
    print(i)
    for j in range(64):
        label.append(label[i-1] + 1)
for j in range(128):
    label.append(8)
y = pd.DataFrame(label)

# 生成初始序列
# label = [0] * 352+[1]*352+[2]*352
# y = pd.DataFrame(label)
#%% train:vaild=3:1
splits = get_splits(y,n_splits=1,valid_size = 0.25,test_size = 0,train_only=False,show_plot=True,check_splits=True,stratify=True, random_state=23, shuffle=True)
#%%

# MLSTM_FCNPlus FCNPlus RNN_FCNPlus
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
model = TSClassifier(X, y.values, splits=splits, path='models', arch="FCNPlus", tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
import time
# 记录训练开始时间
start_time = time.time()
model.fit_one_cycle(25, 0.001)
# 记录训练结束时间
end_time = time.time()
# 计算并打印训练时间
execution_time = end_time - start_time
print(f"训练时间：{execution_time} 秒")
model.export("stage1.pkl")
#%%
df2 = pd.read_csv(r"D:\a_zyc_file\a-zouyuechao\graduation_thesis\experiment\time_series_damage_detection\dataset\MVS_P2_NM_Z\test.csv",header=None)   
# df2 = pd.read_csv(r"D:\a_zyc_file\a-zouyuechao\graduation_thesis\experiment\time_series_damage_detection\dataset\sweden\dataset-test.csv",header=None)

X2, y2 = df2xy(df2, data_cols=None)
# X2=X2.reshape(1280,128,40)
# X2=X2.reshape(264,500,21)
X2=X2.reshape(160,1024,40)
X2=np.transpose(X2,(0,2,1))

label = [0] * 16
for i in range(16, 128, 16):
    for j in range(16):
        label.append(label[i-1] + 1)
for j in range(32):
    label.append(8)
y2 = pd.DataFrame(label)
# label2 = [0] * 88+[1]*88+[2]*88
# y2 = pd.DataFrame(label2)
#%%
from tsai.inference import load_learner

mv_clf = load_learner(r'D:\spyderFile\models\stage1.pkl')
# 记录推理开始时间
start_time1 = time.time()
probas, target, preds = model.get_X_preds(X2[:160], y2.values[:160])
# 记录推理结束时间
end_time1 = time.time()
# 计算并打印推理时间
execution_time1 = end_time1 - start_time1
print(f"推理时间：{execution_time1} 秒")
#%%
print(f'accuracy: {skm.accuracy_score(target.to("cpu").numpy().astype(int), preds.astype(int)):10.6f}')
print(f'precision: {skm.precision_score(target.to("cpu").numpy().astype(int), preds.astype(int), average="weighted"):10.6f}')
print(f'recall: {skm.recall_score(target.to("cpu").numpy().astype(int), preds.astype(int), average="weighted"):10.6f}')
print(f'f1: {skm.f1_score(target.to("cpu").numpy().astype(int), preds.astype(int), average="weighted"):10.6f}')
#%%
import pickle

# 假设 model 是你的模型实例
# 序列化模型
with open(r'C:\Users\zero\models\stage1.pkl', 'wb') as f:
    pickle.dump(model, f)

# 计算文件大小
import os
file_size = os.path.getsize(r'C:\Users\zero\models\stage1.pkl')
file_size_mb = file_size / (1024 * 1024)
print(f"模型文件大小：{file_size_mb} MB")

# 计算参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

params_count = count_parameters(model)
print(f"模型参数数量：{params_count // 1000}K")

#%%
# 展示实际损伤分类和预测分类结果
model.show_results()
#%%
# interp = ClassificationInterpretation.from_learner(model)
# interp.plot_confusion_matrix()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(target.to("cpu").numpy().astype(int), preds.astype(int))

# 选择颜色方案
cmap = 'Blues'  # 或者 'Greys', 'binary'

# 使用 seaborn 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
            annot_kws={"size": 12})  # 设置注释文本的字体大小

# plt.title('Confusion Matrix', fontsize=20)  # 设置标题字体大小
# plt.ylabel('True Label', fontsize=16)  # 设置y轴标签字体大小
# plt.xlabel('Predicted Label', fontsize=16)  # 设置x轴标签字体大小

# 增大刻度标签字体大小
plt.tick_params(axis='both', which='major', labelsize=14)

# 显示颜色条（可选）
plt.colorbar()

plt.show()
#%%
model.feature_importance(save_df_path=r"D:\a_zyc_file\a-zouyuechao\graduation_thesis\experiment\time_series_damage_detection\result\hell.csv")
#%%
model.show_probas()
#%%
tfms  = [None, TSRegression()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)
dls.show_batch(sharey=True)
#%%
nvars = 40
seq_len = 1024
c_out = 1
c_in=40
model=MRNN_FCNPlus(c_in, c_out, seq_len)
# model= ConvTranPlus(c_in, c_out, seq_len)
# model = MLP(nvars, c_out, seq_len)
learn = Learner(dls, model, metrics=accuracy)
learn.save('stage0')
learn.load('stage0')
learn.lr_find()
learn.fit_one_cycle(100, lr_max=0.001)
learn.save('stage1')
learn.recorder.plot_metrics()
learn.show_results()
learn.show_probas()
#%%
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
#%%
import pandas as pd

# 读取原始CSV文件
input_file = 'D:\jupyteNotebookFile\gjemnessund\gjemnessund.csv'  # 替换为你的输入文件路径
df = pd.read_csv(input_file)

# 选择第一列和后6列
# 假设CSV文件有n列，那么后6列的索引范围是 n-6 到 n-1
first_column = df.iloc[:, 0]  # 第一列
last_six_columns = df.iloc[:, -1:]  # 后6列

# 将第一列和后6列合并成一个新的DataFrame
new_df = pd.concat([first_column, last_six_columns], axis=1)

# 保存新的CSV文件
output_file = 'D:\jupyteNotebookFile\gjemnessund\gjemnessund_Univariate.csv'  # 替换为你想要保存的新文件路径
new_df.to_csv(output_file, index=False)

print(f"新文件已保存到 {output_file}")
#%%
import matplotlib.pyplot as plt

# 设置中文字体和负号显示
plt.rcParams['font.family'] = ['Microsoft YaHei']  # 选择一个更美观的中文字体
plt.rcParams['font.size'] = 12  # 设置全局字体大小
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据：训练时间（秒），MAE误差，存储量（MB）和模型名称
train_times = [6.22, 22.48, 28.83, 7.42, 5.50, 12.14, 6.30]  # 训练时间
mae_errors = [0.340, 0.926, 1.081, 0.591, 0.693, 0.620, 0.301]  # MAE误差
memory = [612, 1415, 414, 533, 1906, 1296, 449]  # 存储量，单位MB
models = ['iTransformer', 'Informer', 'Reformer', 'ModernTCN', 'TIDE', 'FreTS', 'RPTM']  # 模型名称

# 定义更鲜艳明亮的颜色方案
colors = [
    [0.2, 0.4, 0.9, 0.9],  # 明亮蓝色
    [0.4, 0.9, 0.4, 0.9],  # 明亮绿色
    [0.9, 0.4, 0.4, 0.9],  # 明亮红色
    [0.6, 0.6, 0.9, 0.9],  # 明亮紫色
    [0.9, 0.6, 0.6, 0.9],  # 明亮橙色
    [0.6, 0.9, 0.6, 0.9],  # 明亮青色
    [0.9, 0.9, 0.6, 0.9]   # 明亮黄色
]
alpha_value = 0.7  # 透明度值

# 绘制散点图，点的大小与存储量成正比，并使用不同的颜色和透明度
for i, model in enumerate(models):
    plt.scatter([train_times[i]], [mae_errors[i]], s=memory[i]*2, color=colors[i], alpha=alpha_value)

# 在每个点旁边标记模型的名称、存储量和训练时间
# for i, model in enumerate(models):
#     annotation = f"{model}\n（{train_times[i]:.2f}s,{memory[i]:.0f}MB）"
#     plt.annotate(annotation, (train_times[i], mae_errors[i]), textcoords="offset points", xytext=(0,0), ha='center', va='center')

# 添加网格线
plt.grid(True)


# 设置图表标题和坐标轴标签
plt.title('效率比较', fontsize=14)  # 调整标题字体大小
plt.xlabel('训练时间（s/epoch）', fontsize=12)  # 调整x轴标签字体大小
plt.ylabel('均方误差', fontsize=12)  # 调整y轴标签字体大小
