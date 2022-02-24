# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from annlp import fix_seed
#
# fix_seed()
#
#
# class Net(nn.Module):
#     def __init__(self, ):
#         super(Net, self).__init__()
#         self.linear = nn.Linear(8, 2, bias=False)
#         self.relu = nn.ReLU()
#         # self.weight = nn.Parameter(1)
#
#     def forward(self, x):
#         x = self.linear(x)
#         x = self.relu(x)
#         return x
#
#
# net = Net()
# batch_size = 32
# y = net(torch.randn(batch_size, 8))
# optimizer = Adam(net.parameters(), lr=0.01)
# criterion = nn.CrossEntropyLoss()
# loss = criterion(y, torch.randint(0, 2, (batch_size,)))
# print(loss)
#
# optimizer.zero_grad()
# loss.backward()
# for p in net.parameters():
#     print(p.grad)
#     # p.grad = p.grad / torch.norm(p.grad, dim=1).unsqueeze(-1)
#     p.grad = p.grad / torch.norm(p.grad)
#     print(p.grad)
#
# import numpy as np
#
# a = [0.0450, 0.0116, 0.0428, 0.0532, -0.0611, -0.0603, -0.0139, 0.0519]
# b = np.array(a) / np.linalg.norm(a)
# print(b)
import pandas as pd

# df1 = pd.read_csv('data/sentiment.csv')
# df2 = pd.read_csv('data/sentiment_hotel.csv')
#
# a = df1['text']
# b = df2['text']
#
# print(len(a))
# print(len(b))
# print(len(a)+len(b))
# print(len(set(pd.concat([a, b]))))

# df = pd.read_csv('data/tc_sentiment.csv')
#
# df['len'] = df['text'].apply(lambda x: len(str(x)))
# df = df[df['len'] > 5]
#
# # print(df.head())
# #
# # a = df.iloc[:, 0:2]
# # a.columns = ['label', 'text']
# df.to_csv('data/tc_sentiment.csv', index=False)

# label = df['label'].unique()
# print(label)
#
# dic = {}
#
# dic = {k: len(dic) for k in label}
#
# print(dic)
#
# df['label'] = df['label'].map(dic)
#
# df.to_csv('data/thucnews.csv', index=False)

import math
import numpy as np


def pick_hard_text_with_entropy(pros, threshold=0.7):
    """
    困难样本检测
    param pros: 文本概率，格式为文本list
    param threshold：信息熵阈值
    """
    if pros is None or len(pros) == 0:
        return False
    result = np.sum(pros * -np.log2(pros))
    max_entropy = np.log2(len(pros))
    if result > threshold * max_entropy:
        return True
    return False


print(pick_hard_text_with_entropy([0.1, 0.3, 0.6]))
