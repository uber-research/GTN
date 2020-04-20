"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import time
import torch
from torch import nn
from torch.autograd import grad as grad_f


torch.backends.cudnn.deterministic = True
test = torch.zeros(128, 32, 32, 32, device='cuda')

n = 100
conv = nn.Conv2d(32, 128, kernel_size=3, padding=1).cuda()
torch.cuda.synchronize()
tstart = time.time()
for _ in range(n):
    x = conv(test)
torch.cuda.synchronize()
print("forward time:", time.time() - tstart)

grad = torch.zeros_like(x)
torch.cuda.synchronize()
tstart = time.time()
for _ in range(n):
    x = conv(test)
    dw, = grad_f(x, conv.weight, grad_outputs=grad, create_graph=True, allow_unused=False)
torch.cuda.synchronize()
print("forward + backward time:", time.time() - tstart)

second_grad = torch.zeros_like(conv.weight)
torch.cuda.synchronize()
tstart = time.time()
for _ in range(n):
    x = conv(test)
    dw, = grad_f(x, conv.weight, grad_outputs=grad, create_graph=True, allow_unused=False)
    dw.backward(second_grad)
torch.cuda.synchronize()
print("forward + backward * 2 time:", time.time() - tstart)
