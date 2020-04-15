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


class AbstractClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._named_parameters = None

    def get_parameters(self):
        if self._named_parameters is None:
            self._named_parameters = list(super().named_parameters())
        return self._named_parameters

    def set_parameters(self, named_parameters):
        self._named_parameters = named_parameters
        for name, value in named_parameters:
            module = self

            while '.' in name:
                module_name, name = name.split('.', 1)
                if isinstance(module, nn.Sequential):
                    module = module._modules[module_name]
                else:
                    module = getattr(module, module_name)

            if name in module._parameters:
                del module._parameters[name]
            setattr(module, name, value)

    def set_buffers(self, named_parameters):
        for name, value in named_parameters:
            module = self

            while '.' in name:
                module_name, name = name.split('.', 1)
                if isinstance(module, nn.Sequential):
                    module = module._modules[module_name]
                else:
                    module = getattr(module, module_name)
            setattr(module, name, value)


class Classifier(AbstractClassifier):
    def __init__(self, input_shape, batch_norm_momentum=0.9, randomize_width=False, use_global_pooling=True):
        super().__init__()

        conv1_size = np.random.randint(32, 128) if randomize_width else 64
        conv2_size = np.random.randint(64, 256) if randomize_width else 128
        fc1_size = np.random.randint(64, 256) if randomize_width else 256
        width = (input_shape[-1] // 4) ** 2
        self.convs = nn.Sequential(
            # Conv1
            WeightNormConv2d(input_shape[0], conv1_size, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(3, 2, padding=1),
            BatchNormMeanOnly(conv1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv2
            WeightNormConv2d(conv1_size, conv2_size, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(3, 2, padding=1),
            BatchNormMeanOnly(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
        )
        if use_global_pooling:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.fc = WeightNormLinear(conv2_size, 10)
        else:
            self.global_pooling = nn.Identity()
            self.fc = nn.Sequential(
                WeightNormLinear(conv2_size * width, fc1_size),
                BatchNormMeanOnly(fc1_size, momentum=batch_norm_momentum),
                nn.LeakyReLU(0.1),
                WeightNormLinear(fc1_size, 10)
            )

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pooling(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class LinearClassifier(AbstractClassifier):
    def __init__(self, input_shape, batch_norm_momentum=0.9, randomize_width=False, use_global_pooling=True):
        super().__init__()

        fc1_size = np.random.randint(64, 256) if randomize_width else 256
        fc2_size = 128
        width = (input_shape[-1]) ** 2
        self.fc = nn.Sequential(
            WeightNormLinear(np.prod(input_shape), fc1_size),
            BatchNormMeanOnly(fc1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormLinear(fc1_size, fc2_size),
            BatchNormMeanOnly(fc2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormLinear(fc2_size, 10)
        )

    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class ClassifierLarger(AbstractClassifier):
    def __init__(self, input_shape, batch_norm_momentum=0.9, randomize_width=False):
        super().__init__()

        conv1_size = np.random.randint(32, 128) if randomize_width else 64
        conv2_size = np.random.randint(64, 256) if randomize_width else 128
        fc1_size = np.random.randint(64, 256) if randomize_width else 256
        width = (input_shape[-1] // 4) ** 2
        self.convs = nn.Sequential(
            # Conv1
            WeightNormConv2d(input_shape[0], conv1_size, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(3, 2, padding=1),
            BatchNormMeanOnly(conv1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv2
            WeightNormConv2d(conv1_size, conv2_size, kernel_size=3, padding=1, stride=1, bias=False),
            nn.MaxPool2d(3, 2, padding=1),
            BatchNormMeanOnly(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
        )
        self.global_pooling = nn.Identity()#nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            WeightNormLinear(conv2_size * width, fc1_size),
            BatchNormMeanOnly(fc1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormLinear(fc1_size, 10)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pooling(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class ClassifierLarger2(AbstractClassifier):
    def __init__(self, input_shape, batch_norm_momentum=0.9, randomize_width=False):
        super().__init__()

        conv1_size = np.random.randint(32, 128) if randomize_width else 64
        conv2_size = np.random.randint(64, 256) if randomize_width else 128
        conv3_size = np.random.randint(64, 256) if randomize_width else 256
        self.convs = nn.Sequential(
            # Conv1
            WeightNormConv2d(input_shape[0], conv1_size, kernel_size=3, padding=1, bias=False),
            BatchNormMeanOnly(conv1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv2
            WeightNormConv2d(conv1_size, conv2_size, kernel_size=3, padding=1, stride=2, bias=False),
            BatchNormMeanOnly(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv2_size, conv2_size, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNormMeanOnly(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv3
            WeightNormConv2d(conv2_size, conv3_size, kernel_size=3, padding=1, stride=2, bias=False),
            BatchNormMeanOnly(conv3_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv3_size, conv3_size, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNormMeanOnly(conv3_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc2 = WeightNormLinear(conv3_size, 10)

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pooling(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class ClassifierLarger3(AbstractClassifier):
    def __init__(self, input_shape, batch_norm_momentum=0.9, randomize_width=False, use_global_pooling=False):
        super().__init__()

        conv1_size = np.random.randint(32, 128) if randomize_width else 64
        conv2_size = np.random.randint(64, 256) if randomize_width else 128
        conv3_size = np.random.randint(64, 512) if randomize_width else 256
        fc1_size = np.random.randint(64, 512) if randomize_width else 256
        width = (input_shape[-1] // 4) ** 2
        self.convs = nn.Sequential(
            # Conv1
            WeightNormConv2d(input_shape[0], conv1_size, kernel_size=3, padding=1, bias=False),
            BatchNormMeanOnly(conv1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv1_size, conv1_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNormMeanOnly(conv1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv2
            WeightNormConv2d(conv1_size, conv2_size, kernel_size=3, padding=1, stride=2, bias=False),
            BatchNormMeanOnly(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv2_size, conv2_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNormMeanOnly(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv3
            WeightNormConv2d(conv2_size, conv3_size, kernel_size=3, padding=1, stride=2, bias=False),
            BatchNormMeanOnly(conv3_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv3_size, conv3_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNormMeanOnly(conv3_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
        )
        if use_global_pooling:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.fc = WeightNormLinear(conv3_size, 10)
        else:
            self.global_pooling = nn.Identity()
            self.fc = nn.Sequential(
                WeightNormLinear(conv3_size * width, fc1_size),
                BatchNormMeanOnly(fc1_size, momentum=batch_norm_momentum),
                nn.LeakyReLU(0.1),
                WeightNormLinear(fc1_size, 10)
            )

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pooling(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class ClassifierLarger4(AbstractClassifier):
    def __init__(self, input_shape, batch_norm_momentum=0.9, randomize_width=False, use_global_pooling=False):
        super().__init__()

        conv1_size = np.random.randint(32, 128) if randomize_width else 64
        conv2_size = np.random.randint(64, 256) if randomize_width else 128
        conv3_size = np.random.randint(64, 512) if randomize_width else 256
        fc1_size = np.random.randint(64, 512) if randomize_width else 256
        width = (input_shape[-1] // 4) ** 2
        self.convs = nn.Sequential(
            # Conv1
            WeightNormConv2d(input_shape[0], conv1_size, kernel_size=3, padding=1, bias=False),
            BatchNorm(conv1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv1_size, conv1_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm(conv1_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv2
            WeightNormConv2d(conv1_size, conv2_size, kernel_size=3, padding=1, stride=2, bias=False),
            BatchNorm(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv2_size, conv2_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv2_size, conv2_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm(conv2_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),

            # Conv3
            WeightNormConv2d(conv2_size, conv3_size, kernel_size=3, padding=1, stride=2, bias=False),
            BatchNorm(conv3_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv3_size, conv3_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm(conv3_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
            WeightNormConv2d(conv3_size, conv3_size, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm(conv3_size, momentum=batch_norm_momentum),
            nn.LeakyReLU(0.1),
        )
        if use_global_pooling:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.fc = WeightNormLinear(conv3_size, 10)
        else:
            self.global_pooling = nn.Identity()
            self.fc = nn.Sequential(
                WeightNormLinear(conv3_size * width, fc1_size),
                BatchNorm(fc1_size, momentum=batch_norm_momentum),
                nn.LeakyReLU(0.1),
                WeightNormLinear(fc1_size, 10)
            )

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pooling(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class Encoder(AbstractClassifier):
    def __init__(self, input_shape, output_size=128):
        super().__init__()

        self.convs = nn.Sequential()
        width = (input_shape[-1] // 4) * (input_shape[-2] // 4)
        self.conv1 = WeightNormConv2d(input_shape[0], 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNormMeanOnly(64)
        self.conv2 = WeightNormConv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNormMeanOnly(128)
        self.fc1 = WeightNormLinear(128 * width, output_size, bias=False)
        self.bn_fc = BatchNormMeanOnly(output_size)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(F.max_pool2d(self.conv1(x), 2)), 0.1)
        x = F.leaky_relu(self.bn2(F.max_pool2d(self.conv2(x), 2)), 0.1)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = F.leaky_relu(self.bn_fc(self.fc1(x)), 0.1)
        return x


# Functional BatchNorm
def fBatchNorm(x, running_mean, running_variance, gamma, beta, eps, momentum, training, mean_only):
    if training:
        if mean_only:
            if len(x.shape) == 2:
                x_mean = torch.mean(x, dim=0)
            else:
                x_mean = torch.mean(x, dim=[0, 2, 3])
        else:
            if len(x.shape) == 2:
                x_var, x_mean = torch.var_mean(x, dim=0)
            else:
                x_var, x_mean = torch.var_mean(x, dim=[0, 2, 3])
        # Linear interpolation: running_mean = running_mean * momentum + x_mean * (1 - momentum)
        running_mean = torch.lerp(x_mean, running_mean, momentum)
        if not mean_only:
            running_variance = torch.lerp(x_var, running_variance, momentum)
    else:
        x_mean = running_mean
        if not mean_only:
            x_var = running_variance

    if len(x.shape) == 2:
        if mean_only:
            normalized = x + (beta - x_mean)
        else:
            normalized = (x - x_mean) * (gamma * (x_var + eps).rsqrt()) + beta
        return normalized, running_mean, running_variance
    elif len(x.shape) == 4:
        if mean_only:
            normalized = x + (beta - x_mean)[..., None, None]
        else:
            normalized = (x - x_mean[..., None, None]) * (gamma[..., None, None] * (x_var[..., None, None] + eps).rsqrt()) + beta[..., None, None]
        return normalized, running_mean, running_variance


class BatchNorm(nn.Module):
    """
    We have to implement batch norm as a weird RNN model. because of gradient checkpointing
    TODO: Undo this hack
    """
    def __init__(self, size, momentum=0.9, eps=1e-5, mean_only=False):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size))
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(size))
        self.mean_only = mean_only
        self.eps = eps
        if not mean_only:
            self.register_buffer("running_variance", torch.ones(size))
            self.gamma = nn.Parameter(torch.ones(size))
        else:
            self.running_variance = None
            self.gamma = None

    def forward(self, x):
        returned = fBatchNorm(x, self.running_mean, self.running_variance, self.gamma, self.bias, self.eps, self.momentum, self.training, self.mean_only)
        x, self.running_mean, self.running_variance = returned
        return x


class BatchNormMeanOnly(BatchNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, mean_only=True)


class BatchNormMeanOnlyOnline(nn.Module):
    def __init__(self, size, momentum=0.1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        if len(x.shape) == 2:
            x_mean = x.mean(dim=0)
        else:
            x_mean = x.mean(dim=[0, 2, 3])

        if len(x.shape) == 2:
            return x - x_mean + self.bias
        elif len(x.shape) == 4:
            return x - x_mean[..., None, None] + self.bias[..., None, None]


class Generator(nn.Module):
    def __init__(self, input_size, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.width = img_shape[-1] // 4
        # TODO: Maybe replace batch norm here?

        self.model = nn.Sequential(
            WeightNormLinear(input_size, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            WeightNormLinear(1024, self.width * self.width * 128, bias=False),
            nn.BatchNorm1d(self.width * self.width * 128),
            nn.LeakyReLU(0.1),
        )
        self.deconv = nn.Sequential(
            Deconv(128, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            Deconv(64, img_shape[0], kernel_size=3),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], 128, self.width, self.width)
        img = self.deconv(img)
        # TODO: Replace constant 4 with a parameter
        # The scalar is meant to allow the generator to match the same range of values as the transformed dataset
        # Since the img_variance for CIFAR10 is 0.25 we need to scale by 2x
        return img * 2


class Deconv(nn.Module):
    def __init__(self, *args, kernel_size, **kwargs):
        super().__init__()
        self.conv = WeightNormConv2d(*args, kernel_size=kernel_size, **kwargs)
        self.pad = nn.ReflectionPad2d((kernel_size - 1) // 2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.pad(x)
        x = self.conv(x)
        return x


def make_weight_norm_layer(base_cls):
    class WeightNorm(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_g = nn.Parameter(torch.norm_except_dim(self.weight, 2, 0).data)

        def forward(self, x):
            x = super().forward(x)
            x = x * (self.weight_g / torch.norm_except_dim(self.weight, 2, 0)).transpose(1, 0)
            return x
    return WeightNorm


class Linear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.kaiming_normal_(self.weight)

from exconv import ExConv2d
class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        if self.bias is None and self.groups == 1 and not getattr(self, "disable_exconv", False):
            return ExConv2d.apply(x, self.weight, self.padding, self.stride, self.dilation, self.groups)
        else:
            return super().forward(x)


WeightNormLinear = make_weight_norm_layer(Linear)
WeightNormConv2d = make_weight_norm_layer(Conv2d)


class ConvNxNBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, groups=1, batch_norm_momentum=0.5):
        super().__init__()
        self.conv = WeightNormConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=False)
        self.bn = BatchNormMeanOnly(out_channels, momentum=batch_norm_momentum)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.1)


class Conv1x1Bn(ConvNxNBn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=1, **kwargs)


class Conv3x3Bn(ConvNxNBn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1, **kwargs)


class SeparableConv3x3Bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm_momentum=0.5):
        super().__init__()
        self.conv = WeightNormConv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.conv2 = WeightNormConv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=1, bias=False)
        self.bn = BatchNormMeanOnly(out_channels, momentum=batch_norm_momentum)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.1)


class SeparableConv1x1Bn(SeparableConv3x3Bn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=1, padding=0, **kwargs)


class MaxPool3x3(nn.MaxPool2d):
    def __init__(self, in_channels=None, out_channels=None, stride=1, batch_norm_momentum=0.5):
        super().__init__(3, stride=stride, padding=1)
        assert in_channels == out_channels
        # TODO: Check shape

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super(MaxPool3x3, self).forward(x)
        return [super(MaxPool3x3, self).forward(a) for a in x]


class AvgPool3x3(nn.AvgPool2d):
    def __init__(self, in_channels=None, out_channels=None, stride=1, batch_norm_momentum=0.5):
        super().__init__(3, stride=stride, padding=1)
        assert in_channels == out_channels
        # TODO: Check shape

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super(AvgPool3x3, self).forward(x)
        return [super(AvgPool3x3, self).forward(a) for a in x]


class Conv3x3MaxPool(Conv3x3Bn):
    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.1)


class SimpleModel(AbstractClassifier):
    def __init__(self, img_shape, arguments, stem_size=64, batch_norm_momentum=0.0, blocks=3):
        super().__init__()
        self.layers = nn.Sequential(
            Conv3x3MaxPool(img_shape[0], stem_size, batch_norm_momentum=batch_norm_momentum)
        )
        img_shape = (stem_size, img_shape[1] // 2, img_shape[2] // 2)

        for block_num in range(blocks):
            if block_num > 0:
                self.layers.add_module(str(len(self.layers) + 1), Conv3x3MaxPool(img_shape[0], img_shape[0] * 2, batch_norm_momentum=batch_norm_momentum))
                img_shape = (img_shape[0] * 2, img_shape[1] // 2, img_shape[2] // 2)

            for arg in arguments:
                op = OP_MAP[arg](img_shape[0], img_shape[0], batch_norm_momentum=batch_norm_momentum)
                self.layers.add_module(str(len(self.layers) + 1), op)

        img_shape = [np.prod(img_shape)]
        self.final = WeightNormLinear(img_shape[0], 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.final(x)
        return F.log_softmax(x, dim=-1)


def sample_model(img_shape, layers=2, blocks=2, encoding=None, seed=None):
    if encoding is None:
        if seed:
            state = np.random.RandomState(seed)
        else:
            state = np.random
        ops = [state.randint(len(OP_MAP)) for i in range(layers)]
    else:
        ops = encoding
    return SimpleModel(img_shape, ops, blocks=blocks), ops

class Generator2(nn.Module):
    def __init__(self, nz, img_shape):
        super(Generator2, self).__init__()
        nz = 100
        ngf = int(64)
        ndf = int(64)
        nc = img_shape[0]
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 1, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input[..., None, None])
        output = nn.Upsample(size=(32, 32), mode='bilinear')(output)
        return output


OP_MAP = [
    Conv3x3Bn,
    Conv1x1Bn,
    SeparableConv3x3Bn,
    SeparableConv1x1Bn,
    MaxPool3x3,
    AvgPool3x3,
]
