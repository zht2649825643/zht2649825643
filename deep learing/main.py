# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
print(train_data.shape, test_data.shape)
# print(test_data['FireplaceQu'][:100])

corr_data = train_data.corr()

print(corr_data['SalePrice'].sort_values(ascending=False))

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

del all_features['PoolQC']

del all_features['MiscFeature']

del all_features['Alley']

del all_features['Fence']   # all_feature.drop("Fence", axis=1)       # 选项2

del all_features['PoolArea']

del all_features['TotRmsAbvGrd']

del all_features['BsmtFinSF2']

del all_features['BsmtHalfBath']

del all_features['MiscVal']

del all_features['LowQualFinSF']

print(all_features.info())

var = 'YrSold'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

#
corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
#
plt.show()

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]

# get data tensor
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_lables = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

print(train_features.shape[1])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(train_features.shape[1], 100)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.10)

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        output = self.relu(output)
        return output


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(train_features.shape[1], 100)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.10)

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        # output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        return output


loss_fn = nn.MSELoss()


def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss_fn(clipped_preds.log(), labels.log()).mean())
    return rmse.item()


def train(net, train_features, train_lables, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    dataset = torch.utils.data.TensorDataset(train_features, train_lables)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            net.train()
            X = net(X.float())
            l = loss_fn(X.float(), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_lables))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# train(net, train_data, train_lables, num_epochs, learning_rate, weight_decay, batch_size)

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)

        for param in net.parameters():
            nn.init.normal_(param, mean=0, std=0.01)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        #         if i == 0:
        #             d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
        #                          range(1, num_epochs + 1), valid_ls,
        #                          ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


num_epochs = 150
learning_rate = 0.01
weight_decay = 0.01
batch_size = 64
train_ls = []
test_ls = []
test_labels = None
k = 5
net = Net1()
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)
train_ls, _ = train(net, train_features, train_lables, test_features, None, num_epochs, learning_rate, weight_decay,
                    batch_size)
# k_fold(k, train_features, train_lables, num_epochs, learning_rate, weight_decay, batch_size)

# print(train_ls[180:])
#
x = np.array(range(len(train_ls)))

y = train_ls

plt.plot(x, y)

plt.show()

# print(len(x), len(output))
#
preds = net(test_features).detach().numpy().astype(int)

test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])

submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

submission.to_csv('./submission.csv', index=False)
