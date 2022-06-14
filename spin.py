import torch
a = torch.ones([1024, 2048]).to("cuda:3")
while True:
  a = a * a
