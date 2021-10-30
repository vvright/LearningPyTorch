import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
sampling = multinomial.Multinomial(1, fair_probs).sample()
print(fair_probs)
print(sampling)

counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
print(cum_counts)
estimates = cum_counts /cum_counts.sum(dim=1,keepdims=True)
print(estimates)