"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
from torch import nn
import numpy as np
from gradient_helpers import gradient_checkpointing


def test_gradient(device="cuda"):
    n_iter = 4
    input_variable = nn.Parameter(torch.randn(10, device=device) + 1)
    loop_variable = nn.Parameter(torch.randn(10, device=device) + 1)
    after_loop_variable = nn.Parameter(torch.randn(10, device=device) + 1)
    variables = input_variable, loop_variable, after_loop_variable

    def body(state):
        it, a = state
        a = (a) * loop_variable
        return it + 1, a

    initial_state = torch.randn_like(input_variable) + 1
    state = initial_state * input_variable
    it, state = gradient_checkpointing((torch.as_tensor(0), state), body, n_iter, block_size=0)
    assert it == n_iter
    state = state * after_loop_variable
    state.sum().backward()
    state1 = state.detach()

    grads = [v.grad.clone().detach() for v in variables]

    for v in variables:
        v.grad.zero_()

    state = initial_state * input_variable
    it, state = gradient_checkpointing((torch.as_tensor(0), state), body, n_iter, block_size=0)
    assert it == n_iter
    state = state * after_loop_variable
    state.sum().backward()

    # Make sure that evaluation is deterministic
    assert np.allclose(state.detach().cpu(), state1.detach().cpu())
    for g, v in zip(grads, variables):
        assert np.allclose(g.detach().cpu(), v.grad.detach().cpu())

    # Now check checkpointing
    for block_size in [1, 100]:
        for v in variables:
            v.grad.zero_()

        state = initial_state * input_variable
        it, state = gradient_checkpointing((torch.as_tensor(0), state), body, n_iter, block_size=block_size)
        assert it == n_iter
        state = state * after_loop_variable
        state.sum().backward()

        assert np.allclose(state.detach().cpu(), state1.detach().cpu())
        for g, v in zip(grads, variables):
            assert np.allclose(g.detach().cpu(), v.grad.detach().cpu())
    print("Test Successful")


if __name__ == "__main__":
    import fire
    fire.Fire(test_gradient)
