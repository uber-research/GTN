"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import nest
from torch.utils.checkpoint import get_device_states, set_device_states


def detach_variable(v):
    if isinstance(v, torch.Tensor):
        new_v = v.detach()
        if v.requires_grad and torch.Tensor.is_floating_point(v):
            new_v.requires_grad = True
        return new_v
    return v


def compute_added_grads(a, b):
    if b is not None:
        return a + b
    return a


class SurrogateLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss_fn, *state):
        ctx.save_for_backward(*state)
        ctx.loss_fn = loss_fn
        return tuple(state)

    @staticmethod
    def backward(ctx, *grad_output):
        detached_inputs = [detach_variable(v) for v in ctx.saved_tensors]
        with torch.enable_grad():
            loss = ctx.loss_fn(detached_inputs)
        extra_grads = torch.autograd.grad(loss, [d for d in detached_inputs if d.requires_grad], allow_unused=True)
        extra_grads = iter(extra_grads)
        result = (None,) + tuple(compute_added_grads(a, next(extra_grads)) if d.requires_grad else a for a, d in zip(grad_output, detached_inputs))
        try:
            next(extra_grads)
            raise ValueError("extra_grads left")
        except StopIteration:
            return result


class GradientCheckpointBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, structure, block_size, body_fn, *state):
        ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*state)
        with torch.enable_grad():
            ctx.devices = [s.device for s in state]
            cpu_state = nest.map_structure(lambda x: x.to('cpu', non_blocking=True), state)
        ctx.save_for_backward(*cpu_state)
        ctx.structure = structure
        ctx.block_size = block_size
        ctx.body_fn = body_fn
        state = nest.pack_sequence_as(ctx.structure, state)
        ctx.fwd_cpu_state = torch.get_rng_state()
        with torch.no_grad():
            for _ in range(block_size):
                state = body_fn(state)
        state = nest.flatten(state)
        return tuple(state)

    @staticmethod
    def backward(ctx, *grad_output):
        with torch.enable_grad():
            detached_inputs = [detach_variable(v.to(device, non_blocking=True)) for v, device in zip(ctx.saved_tensors, ctx.devices)]
            state = nest.pack_sequence_as(ctx.structure, detached_inputs)
            next_state = state
            rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                torch.set_rng_state(ctx.fwd_cpu_state)
                set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                for _ in range(ctx.block_size):
                    next_state = ctx.body_fn(next_state)
        next_state = nest.flatten(next_state)

        next_state, grad_output = zip(*[sg for sg in zip(next_state, grad_output) if sg[0].requires_grad])
        torch.autograd.backward(next_state, grad_output)

        return (None, None, None) + tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                                          for inp in detached_inputs)


def gradient_checkpointing(state, body_fn, total_iterations, block_size=16, checkpoint_last_iter=True):
    """
    checkpoint_last_iter: Indicates rather we checkpoint the final state (useful if more operations are done after)
    """
    if total_iterations == 0:
        return state
    if block_size == 0:
        # Skip gradient_checkpointing
        for _ in range(total_iterations):
            state = body_fn(state)
        return state
    structure = nest.map_structure(lambda x: None, state)
    state = nest.flatten(state)
    current_iteration = 0
    if total_iterations > block_size:
        for _ in range(int(total_iterations // block_size - 1)):
            state = GradientCheckpointBlock.apply(structure, block_size, body_fn, *state)
            current_iteration += block_size

    if checkpoint_last_iter:
        state = GradientCheckpointBlock.apply(structure, total_iterations - current_iteration, body_fn, *state)
        current_iteration += total_iterations - current_iteration
        state = nest.pack_sequence_as(structure, state)
    else:
        state = nest.pack_sequence_as(structure, state)
        for _ in range(current_iteration, total_iterations):
            state = body_fn(state)
    return state


class GradientCheckpointWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, body_fn, *state):
        ctx.save_for_backward(*[s if isinstance(s, torch.Tensor) else None for s in state])
        ctx.extras = [s if not isinstance(s, torch.Tensor) else None for s in state]
        ctx.body_fn = body_fn
        with torch.no_grad():
            state = body_fn(*state)
        return tuple([s for s in state if isinstance(s, torch.Tensor)])

    @staticmethod
    def backward(ctx, *grad_output):
        with torch.enable_grad():
            detached_inputs = [e if v is None else detach_variable(v) for v, e in zip(ctx.saved_tensors, ctx.extras)]
            next_state = ctx.body_fn(*detached_inputs)

        grad_output = [g for s, g in zip(next_state, grad_output) if isinstance(s, torch.Tensor) and s.requires_grad]
        next_state = [s for s in next_state if isinstance(s, torch.Tensor) and s.requires_grad]
        grad_inputs = [s for s in detached_inputs if isinstance(s, torch.Tensor) and s.requires_grad]
        grads = torch.autograd.grad(next_state, grad_inputs, grad_output, create_graph=True, allow_unused=True)
        grads = iter(grads)
        return (None,) + tuple(next(grads) if isinstance(inp, torch.Tensor) and inp.requires_grad else None
                               for inp in detached_inputs)
