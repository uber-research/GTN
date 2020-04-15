"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch
from torch import nn


class BaseOptimizer(nn.Module):
    def initial_state(self, params):
        return tuple(self._initial_state_impl(p) for p in params)

    def _initial_state_impl(self, param):
        raise NotImplementedError()

    def compute_update(self, it, param, grad, state):
        raise NotImplementedError()

    def forward(self, it, params, grads, state=None):
        new_params = [self.compute_update(it, p, s) for p, s in zip(params, state)]
        return new_params, state


class SGD(BaseOptimizer):
    def __init__(self, init_lr, init_momentum, num_inner_iterations=None):
        super().__init__()
        if num_inner_iterations is None:
            self.log_lr = nn.Parameter(torch.as_tensor(np.log(init_lr + 1e-6)))
            self.log_momentum = nn.Parameter(torch.as_tensor(np.log(init_momentum + 1e-6)))
        else:
            self.log_lr = nn.Parameter(torch.full((num_inner_iterations,), np.log(init_lr + 1e-6)))
            self.log_momentum = nn.Parameter(torch.full((num_inner_iterations,), np.log(init_momentum + 1e-6)))

    @property
    def lr(self):
        return torch.exp(self.log_lr)

    @property
    def momentum(self):
        return torch.exp(self.log_momentum)

    def _initial_state_impl(self, param):
        return torch.zeros_like(param)

    @staticmethod
    def compute_update(it, param, grad, state, momentum, lr):
        if state is None:
            state = grad
        state = state * momentum + grad

        return param - lr * state, state

    def forward(self, it, params, grads, state=None):
        momentum = self.momentum[it]
        lr = self.lr[it]
        new_params, state = zip(*[self.compute_update(it, p, g, s, momentum, lr) for p, g, s in zip(params, grads, state)])
        return new_params, state


def inv_sigmoid(x):
    return - np.log(1. / x - 1)


class Adam(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta2_logit = nn.Parameter(torch.as_tensor(inv_sigmoid(0.5)))
        self.log_eps = nn.Parameter(torch.as_tensor(np.log(1e-8)))

    @property
    def beta2(self):
        return torch.sigmoid(self.beta2_logit)

    @property
    def eps(self):
        return torch.exp(self.log_eps)

    def _initial_state_impl(self, param):
        return (torch.zeros_like(param), torch.zeros_like(param))

    @staticmethod
    def compute_update(param, grad, state, beta1, beta2, lr, t, eps, eps1, eps2):
        state1, state2 = state
        state1 = state1 * beta1 + grad * (1.0 - beta1)
        state2 = state2 * beta2 + grad * grad * (1.0 - beta2)
        lr2 = lr * torch.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

        state = state1, state2
        return param - lr2 * state1 / (torch.sqrt(state2 + eps1) + eps + eps2), state

    def forward(self, it, params, grads, state=None):
        beta1 = self.momentum[it]
        lr = self.lr[it]

        beta2 = self.beta2
        eps = self.eps

        eps1, eps2 = 1e-5, 1e-5

        # This seems to be a huge bottleneck right now
        t = torch.as_tensor(it + 1).to(torch.float).to(lr.device)
        new_params, state = zip(*[self.compute_update(p, g, s, beta1, beta2, lr, t, eps, eps1, eps2) for p, g, s in zip(params, grads, state)])
        return new_params, state


class RMSProp(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_logit = nn.Parameter(torch.as_tensor(inv_sigmoid(0.5)))
        self.log_eps = nn.Parameter(torch.as_tensor(np.log(1e-8)))

        self.compute_update = torch.jit.trace(RMSProp.compute_update_,
                                              (torch.as_tensor(0), torch.rand(3), torch.rand(3), (torch.rand(3), torch.rand(3)), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0))
                                              )

    @property
    def decay(self):
        return torch.sigmoid(self.decay_logit)

    @property
    def eps(self):
        return torch.exp(self.log_eps)

    def _initial_state_impl(self, param):
        return (torch.zeros_like(param), torch.zeros_like(param))

    @staticmethod
    def compute_update_(it, param, grad, state, decay, momentum, lr, eps):
        state1, state2 = state
        # state1 = state1 * decay + grad * grad * (1.0 - decay)
        state1 = torch.lerp(grad ** 2, state1, decay)

        grad = grad * (state1 + eps + 1e-8).rsqrt()

        new_param, state2 = SGD.compute_update(it, param, grad, state2, momentum, lr)
        state = state1, state2
        return new_param, state

    def forward(self, it, params, grads, state=None):
        decay = self.decay
        eps = self.eps
        momentum = self.momentum[it]
        lr = self.lr[it]
        new_params, state = zip(*[self.compute_update(it, p, g, s, decay, momentum, lr, eps) for p, g, s in zip(params, grads, state)])
        return new_params, state
