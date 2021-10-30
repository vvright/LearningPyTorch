import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# generating the dataset
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# reading the dataset
def load_array(data_arrays, batch_size, is_train=True):  # @save
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

# defining the model
net = nn.Sequential(nn.Linear(2, 1))

# initializing model parameters
net[0].weight.data.normal_(0, 0.1)
net[0].bias.data.fill_(0)

# defining the loss function
loss = nn.MSELoss()

# defining the optimization algorithm
trainer = torch.optm.SGD(net.parameters(), lr=0.03)

# training
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch:{epoch + 1},loss{l:f}')

w = net[0].weight.data
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
