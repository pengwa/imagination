import torch
a = torch.ones([1024, 2048]).to("cuda:0")
while True:
  a = a * a
