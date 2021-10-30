import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(x.grad == 4 * x)

#对标量求导

x.grad.zero_()
y = x.sum()
print('y = x.sum()')
print(y)
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
print('y.sum()')
print(y.sum())
y.sum().backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)
