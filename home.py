from torch.utils import data
from d2l import torch as d2l
from torch import nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import os
import tarfile
import zipfile
import requests

# 创建字典
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 下载数据集默认保存在../data


def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    # 当assert为False时 返回，后边句子
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    # 允许原文件已经存在
    # 目标目录已存在的情况下不会触发异常。
    os.makedirs(cache_dir, exist_ok=True)
    # split作用是将所有字符串用/分开，加入列表，并取出最后的 文件名 + 后缀
    # 'kaggle_house_pred_train.csv'
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):  # 是否存在当前目录
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                # 1048576 = 1024 * 1024 单位字节，就是1M
                data = f.read(1048576)
                if not data:
                    break

                sha1.update(data)  # 要对哪个字符串进行加密，就放这里
        if sha1.hexdigest() == sha1_hash:  # (前)拿到加密字符串,判断是否是我要读的字符串
            return fname  # 命中缓存
    # 如果不存在该文件，就从网上下载
    print(f'正在从{url}下载{fname}...')
    # 利用requests请求下载文件
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

# ⼀个将下载并解压缩⼀个zip或tar⽂件

# folder是当前目录下的某个文件


def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    # kaggle_house_pred_train.csv去掉文件名，返回目录，
    # 例如print(os.path.dirname("E:/Read_File/read_yaml.py"))
    # 结果：E:/Read_File
    base_dir = os.path.dirname(fname)
    # 分离文件名与扩展名
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    # extractall函数作用为解压压缩包中的所有文件至指定文件夹
    fp.extractall(base_dir)  # 解压到这个文件名
    return os.path.join(base_dir, folder) if folder else data_dir


# 是将使⽤的所有数据集从DATA_HUB下载到缓存⽬录中
def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


# 将使⽤pandas读⼊并处理数据,清洗数据
# 网站提供了hash值，用来检验文件是否被篡改
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 读取相对目录下得csv文件
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data.shape)  # (1460, 81)
print(test_data.shape)  # (1459, 80)
# 位置索引iloc看看前四个和最后两个特征的前四个数据，以及相应标签（房价）
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# 纵向表堆叠
# 去掉id避免过拟合训练时记住id，concat默认是纵向连接
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若⽆法获得测试数据，则可根据训练数据计算均值和标准差
# 找出所有数字列的下标，（内）这些列设置为True，else False，（外）找索引
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 每一列求平均值对于nan跳过不处理
# 标准化数据
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 检测与处理异常值将nan数据改为均值0，这一步放在前边就是设置为x.mean,将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 哑变量处理,处理离散值
# “MSZoning”包含值“RL”和“Rm”。我
# 们将创建两个新的指⽰器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。根据独热编码，如果
# “MSZoning”的原始值为“RL”，则：“MSZoning_RL”为1，“MSZoning_RM”为0。pandas软件包会⾃动为
# 我们实现这⼀点。
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指⽰符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape  # 列会增多 (2919, 331)

# 通过values属性，我们可以从pandas格式中提
# 取NumPy格式，并将其转换为张量表⽰⽤于训练。
n_train = train_data.shape[0]
train_features = torch.tensor(
    all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(
    all_features[n_train:].values, dtype=torch.float32)
# 升维1变2,将一维矩阵变成二维矩阵
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
# 均方损失
loss = nn.MSELoss()
# 判断维数以进行线性仿射变换
in_features = train_features.shape[1]

# 线性模型提供了⼀种健全性检查,如果⼀切顺利，线性模型将作为基线（baseline）模型，让我们直观地知道最
# 好的模型有超出简单的模型多少


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    # 将取值限制到1-inf
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # 利用相对误差代替绝对误差，并取log
    # 均方根误差
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()  # 返回数字


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    # 快速构建dataset，传入训练集和标签，进行解包
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 加载数据集
    train_iter = load_array((train_features, train_labels), batch_size)
    # Adam优化器的主要吸引⼒在于它对初始学习率不那么敏感
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:  # test_labels为列表，非空
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls

# K折交叉验证

# 返回训练集和测试集


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # 注意这里整除，不是注释
    # X_train, y_train被多次赋值，因此需要提前定义
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            # 按列进行合并↓
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # 每次调用增加一列，就加上最后一列
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # 画出图像
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    # 返回平均损失
    return train_l_sum / k, valid_l_sum / k


# k折交叉检验调超参数
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

# 最后使用真正的测试集进行验证，保存本地代替提交


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    # 解包numpy加入列，并且第一列加上编号
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # print("test_data['SalePrice']",test_data['SalePrice'])
    # print('\n')
    # print("test_data['Id']",test_data['Id'])
    # print('\n')
    # 经过测试，这个横向合并共同的第一列标号只留一个
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    print("submission", submission)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
