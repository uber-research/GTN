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
import torch.nn.functional as F
from models import AbstractClassifier, Linear, BatchNorm, Conv2d


def compute_vertex_channels(input_channels, output_channels, matrix):
    """Computes the number of channels at every vertex.

    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.

    Args:
        input_channels: input channel count.
        output_channels: output channel count.
        matrix: adjacency matrix for the module (pruned by model_spec).

    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


def truncate(inputs, channels):
    """Slice the inputs to channels if necessary."""
    input_channels = inputs.shape[1]

    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs.narrow(1, 0, channels)


class NASBenchClassifier(AbstractClassifier):
    def __init__(self, input_shape, adj, vertices, num_stacks=3, num_modules_per_stack=3, stem_filter_size=128):
        super().__init__()

        self.stem_conv = Conv3x3BnRelu(input_shape[0], stem_filter_size)

        channels = stem_filter_size
        input_channels = channels
        stacks = []
        for stack_num in range(num_stacks):
            stack_layers = []
            if stack_num > 0:
                stack_layers.append(nn.MaxPool2d(2, 2, padding=1))
                channels *= 2

            for module_num in range(num_modules_per_stack):
                stack_layers.append(NASModule(input_channels, channels, adj, vertices))
                input_channels = channels
            stacks.append(nn.Sequential(*stack_layers))
        self.stacks = nn.Sequential(*stacks)

        self.fc = Linear(channels, 10)
        # TODO: Maybe normalize

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stacks(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class NASModule(nn.Module):
    def __init__(self, input_channels, channels, matrix, vertices):
        super().__init__()
        self.matrix = matrix
        self.channels = channels
        self.vertices = vertices
        self.build_module(matrix, vertices, input_channels, channels)

    def build_module(self, matrix, vertices, input_channels, channels):
        module_layers = []

        num_vertices = np.shape(matrix)[0]

        vertex_channels = compute_vertex_channels(input_channels, channels, matrix)

        for t in range(1, num_vertices - 1):
            module_layers.append(OP_MAP[vertices[t]](vertex_channels[t], vertex_channels[t]))

            if matrix[0, t]:
                module_layers[-1].projection = Conv1x1BnRelu(input_channels, vertex_channels[t])

        if np.sum(self.matrix[1:, num_vertices - 1]) == 0 or self.matrix[0, num_vertices - 1]:
            self.output_projection = Conv1x1BnRelu(input_channels, channels)

        self.module_layers = nn.ModuleList(module_layers)

    def forward(self, inputs):
        num_vertices = np.shape(self.matrix)[0]
        input_channels = inputs.shape[1]

        vertex_channels = compute_vertex_channels(input_channels, self.channels, self.matrix)
        tensors = [inputs]

        final_concat_in = []
        for t in range(1, num_vertices - 1):
            add_in = [truncate(tensors[src], vertex_channels[t]) for src in range(1, t) if self.matrix[src, t]]

            if self.matrix[0, t]:
                add_in.append(self.module_layers[t - 1].projection(tensors[0]))

            if len(add_in) == 1:
                add_in = add_in[0]
            else:
                add_in = sum(add_in[1:], add_in[0])

            tensors.append(self.module_layers[t - 1](add_in))
            if self.matrix[t, num_vertices - 1]:
                final_concat_in.append(tensors[t])

        if not final_concat_in:
            output = self.output_projection(tensors[0])
        else:
            if len(final_concat_in) == 1:
                output, = final_concat_in
            else:
                output = torch.cat(final_concat_in, 1)

            if self.matrix[0, num_vertices - 1]:
                output = output + self.output_projection(tensors[0])

        return output


class ConvNxNBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = weight_norm(Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.bn = nn.BatchNorm2d(out_channels, momentum=1.0 - 0.997)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x)


class Conv1x1BnRelu(ConvNxNBnRelu):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1)


class Conv3x3BnRelu(ConvNxNBnRelu):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1)


class MaxPool3x3(nn.MaxPool2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(3, stride=1, padding=1)
        assert in_channels == out_channels
        # TODO: Check shape


OP_MAP = {
    'identity': nn.Identity,
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3,
}
