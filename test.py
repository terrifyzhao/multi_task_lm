import torch
import torch.nn as nn
from torch.optim import Adam
from annlp import fix_seed

fix_seed()


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.linear = nn.Linear(8, 2, bias=False)
        self.relu = nn.ReLU()
        # self.weight = nn.Parameter(1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


net = Net()
batch_size = 32
y = net(torch.randn(batch_size, 8))
optimizer = Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
loss = criterion(y, torch.randint(0, 2, (batch_size,)))
print(loss)

optimizer.zero_grad()
loss.backward()
for p in net.parameters():
    print(p.grad)
    # p.grad = p.grad / torch.norm(p.grad, dim=1).unsqueeze(-1)
    p.grad = p.grad / torch.norm(p.grad)
    print(p.grad)

import numpy as np

a = [0.0450, 0.0116, 0.0428, 0.0532, -0.0611, -0.0603, -0.0139, 0.0519]
b = np.array(a) / np.linalg.norm(a)
print(b)
