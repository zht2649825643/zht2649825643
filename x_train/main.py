# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from model import Model
from net import Net
from path import MODEL_PATH
from torch.optim import Adam


'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()


# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
device = torch.device(device)

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

cnn = Net().to(device)
optimizer = Adam(cnn.parameters(), lr=0.00001)  # 选用AdamOptimizer
# optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

'''
实现自己的网络机构
'''
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
net = Net().to(device)
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

'''
dataset.get_step() 获取数据的总迭代次数

'''


def eval(model, x_test, y_test):
    cnn.eval()
    batch_eval = model.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        outputs = cnn(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len


best_score = 0
best_accuracy = 0
loss_list = []
acc_list = []
for step in range(args.EPOCHS):
    cnn.train()
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.long().to(device)
    # print(y_train.shape)

    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)
    x_val = x_val.float().to(device)
    y_val = y_val.long().to(device)
    print(y_train)

    outputs = cnn(x_train)
    # print(outputs.shape)
    print(outputs)
    _, prediction = torch.max(outputs.data, 1)
    print(prediction)

    optimizer.zero_grad()
    # print(x_train.shape, outputs.shape, y_train.shape)
    loss = loss_fn(outputs, y_train)
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    print(loss)

    '''
    实现自己的模型保存逻辑
    '''
    train_accuracy = eval(model, x_val, y_val)
    print(train_accuracy)
    acc_list.append(train_accuracy)
    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (step, best_accuracy))

    print(str(step + 1) + "/" + str(args.EPOCHS))