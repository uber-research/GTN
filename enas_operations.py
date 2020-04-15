"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import WeightNormConv2d as Conv2d, WeightNormLinear as Linear
from models import BatchNorm as _BatchNorm


def ReLU():
    return nn.LeakyReLU(0.1)


def BatchNorm(size, momentum=0.0):
    return _BatchNorm(size, momentum=0.0)


def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id + 1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        # x.div_(drop_path_keep_prob)
        # x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        self.multi_adds = 0
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]

        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        # previous reduction cell
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.relu = ReLU()
            self.preprocess_x = FactorizedReduce(c[0], channels, affine)
            x_out_shape = [hw[1], hw[1], channels]
            self.multi_adds += 1 * 1 * c[0] * channels * hw[1] * hw[1]
        elif c[0] != channels:
            self.preprocess_x = nn.Sequential(
                Conv1x1Bn(c[0], channels)
            )
            x_out_shape = [hw[0], hw[0], channels]
            self.multi_adds += 1 * 1 * c[0] * channels * hw[1] * hw[1]
        if c[1] != channels:
            self.preprocess_y = nn.Sequential(
                Conv1x1Bn(c[1], channels)
            )
            y_out_shape = [hw[1], hw[1], channels]
            self.multi_adds += 1 * 1 * c[1] * channels * hw[1] * hw[1]

        self.out_shape = [x_out_shape, y_out_shape]

    def forward(self, s0, s1):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1)
        return [s0, s1]


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()
        self.relu1 = ReLU()
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = BatchNorm(128)
        self.relu2 = ReLU()
        self.conv2 = Conv2d(128, 768, 2, bias=False)
        self.bn2 = BatchNorm(768)
        self.relu3 = ReLU()
        self.classifier = Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class ConvNxNBn(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, affine=True):
        super().__init__()

        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                ReLU(),
                Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                BatchNorm(C_out)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                ReLU(),
                Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=False),
                BatchNorm(C_out),
                ReLU(),
                Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False),
                BatchNorm(C_out),
            )

    def forward(self, x):
        return self.ops(x)


class Conv1x1Bn(ConvNxNBn):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


class SeparableConv3x3Bn(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            ReLU(),
            Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, stride=stride, bias=False),
            Conv2d(in_channels, in_channels, kernel_size=1, padding=0, groups=1, bias=False),
            BatchNorm(out_channels, momentum=0.5),
            ReLU(),
            Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
            Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=1, bias=False),
            BatchNorm(out_channels, momentum=0.5),
        )


class SeparableConv1x1Bn(SeparableConv3x3Bn):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1, padding=0)


# TODO: Remove affine option (or add it to batchnorm)
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = ReLU()
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = BatchNorm(C_out)

    def forward(self, x):
        x = self.relu(x)
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class FinalCombine(nn.Module):
    def __init__(self, layers, out_hw, channels, concat, affine=True):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat = concat
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        self.multi_adds = 0
        for i in concat:
            hw = layers[i][0]
            if hw > out_hw:
                assert hw == 2 * out_hw and i in [0, 1]
                self.concat_fac_op_dict[i] = len(self.ops)
                self.ops.append(FactorizedReduce(layers[i][-1], channels, affine))
                self.multi_adds += 1 * 1 * layers[i][-1] * channels * out_hw * out_hw

    def forward(self, states):
        for i in self.concat:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i])
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out


OPERATIONS = {
    0: SeparableConv3x3Bn,  # 3x3
    1: SeparableConv3x3Bn,  # 5x5
    2: nn.AvgPool2d,  # 3x3
    3: nn.MaxPool2d,  # 3x3
    4: Identity,
}


OPERATIONS_large = {
    5: Identity,
    6: ConvNxNBn,  # 1x1
    7: ConvNxNBn,  # 3x3
    8: ConvNxNBn,  # 1x3 + 3x1
    9: ConvNxNBn,  # 1x7 + 7x1
    10: nn.MaxPool2d,  # 2x2
    11: nn.MaxPool2d,  # 3x3
    12: nn.MaxPool2d,  # 5x5
    13: nn.AvgPool2d,  # 2x2
    14: nn.AvgPool2d,  # 3x3
    15: nn.AvgPool2d,  # 5x5
}

OPERATIONS_large_name = {
    5: "identity",
    6: "conv 1x1",
    7: "conv 3x3",
    8: "conv 1x3+3x1",
    9: "conv 1x7+7x1",
    10: "max_pool 2x2",
    11: "max_pool 3x3",
    12: "max_pool 5x5",
    13: "avg_pool 2x2",
    14: "avg_pool 3x3",
    15: "avg_pool 5x5",
}
