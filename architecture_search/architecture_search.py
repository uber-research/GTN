"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pickle
import json
import time
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import torchvision
import torchvision.datasets as datasets
import horovod.torch as hvd
from models import Classifier, Generator, Encoder, sample_model
import inner_optimizers
from gradient_helpers import SurrogateLoss, gradient_checkpointing
import nest
import models
from train_cgtn import main, Learner
import enas_micro_models as enas
import tabular_logger as tlogger


class AutoML(nn.Module):
    def __init__(self, generator, optimizers, initial_batch_norm_momentum=0.9):
        super().__init__()
        self.generator = generator
        self.optimizers = torch.nn.ModuleList(optimizers)
        self.batch_norm_momentum_logit = nn.Parameter(torch.as_tensor(inner_optimizers.inv_sigmoid(0.9)))

    @property
    def batch_norm_momentum(self):
        return torch.sigmoid(self.batch_norm_momentum_logit)

    def sample_learner(self, input_shape, device, allow_nas=False, learner_type="base",
                       iteration_maps_seed=False, iteration=None, deterministic=False, iterations_depth_schedule=100, randomize_width=False):
        matrix = np.array([[0, 1, 1, 0, 1, 0, 1],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0]])
        vertices = ['input',
                    'conv3x3-bn-relu',
                    'conv3x3-bn-relu',
                    'conv3x3-bn-relu',
                    'conv3x3-bn-relu',
                    'conv1x1-bn-relu',
                    'output']

        if iteration_maps_seed:
            iteration = iteration - 1
            encoding = [iteration % 6, iteration // 6]
        else:
            encoding = None

        if learner_type == "enas":
            layers = 3
            nodes = 5
            channels = 32
            arch = enas.sample_enas(nodes)
            encoding = json.dumps(enas.encoding_to_dag(arch))
            tlogger.info("Architecture", arch)
            tlogger.info(f"nodes={nodes}, layers={layers}, channels={channels}")
            model = enas.NASNetworkCIFAR([], 10, use_aux_head=False, steps=1,
                                         keep_prob=1.0, drop_path_keep_prob=None,
                                         nodes=nodes,
                                         arch=arch,
                                         channels=channels,
                                         layers=layers  # N
                                         )
        elif learner_type == "enas_1":
            layers = 1
            nodes = 1
            channels = 32
            arch = enas.sample_enas(nodes)
            encoding = json.dumps(enas.encoding_to_dag(arch))
            tlogger.info("Architecture", arch)
            tlogger.info(f"nodes={nodes}, layers={layers}, channels={channels}")
            model = enas.NASNetworkCIFAR([], 10, use_aux_head=False, steps=1,
                                         keep_prob=1.0, drop_path_keep_prob=None,
                                         nodes=nodes,
                                         arch=arch,
                                         channels=channels,
                                         layers=layers  # N
                                         )
        elif learner_type == "enas_fixed":
            layers = 1
            nodes = 5
            channels = 32
            # Use a random fixed architecture
            arch = [[1, 6, 1, 5, 0, 13, 2, 7, 3, 8, 3, 5, 4, 9, 1, 9, 4, 9, 2, 7], [1, 14, 0, 8, 0, 8, 1, 7, 0, 14, 3, 5, 4, 14, 0, 14, 2, 9, 1, 13]]
            encoding = json.dumps(enas.encoding_to_dag(arch))
            tlogger.info("Architecture", arch)
            tlogger.info(f"nodes={nodes}, layers={layers}, channels={channels}")
            model = enas.NASNetworkCIFAR([], 10, use_aux_head=False, steps=1,
                                         keep_prob=1.0, drop_path_keep_prob=None,
                                         nodes=nodes,
                                         arch=arch,
                                         channels=channels,
                                         layers=layers  # N
                                         )
        elif learner_type == "sampled":
            layers = min(4, max(0, iteration // iterations_depth_schedule))
            model, encoding = sample_model(input_shape, layers=layers, encoding=encoding, blocks=2,
                                           seed=iteration if deterministic else None, batch_norm_momentum=0)
            tlogger.record_tabular("encoding", encoding)
        elif learner_type == "sampled4":
            model, encoding = sample_model(input_shape, layers=4, encoding=encoding, seed=iteration if deterministic else None, batch_norm_momentum=0)
            tlogger.record_tabular("encoding", encoding)
        elif learner_type == "base":
            model = Classifier(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_fc":
            model = Classifier(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width, use_global_pooling=False)
        elif learner_type == "linear":
            model = models.LinearClassifier(input_shape)
        elif learner_type == "base_larger":
            model = models.ClassifierLarger(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_larger2":
            model = models.ClassifierLarger2(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_larger3":
            model = models.ClassifierLarger3(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_larger3_global_pooling":
            model = models.ClassifierLarger3(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width, use_global_pooling=True)
        elif learner_type == "base_larger4_global_pooling":
            model = models.ClassifierLarger4(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width, use_global_pooling=True)
        elif learner_type == "base_larger4":
            model = models.ClassifierLarger4(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width, use_global_pooling=False)
        else:
            raise NotImplementedError()

        return Learner(model=model.to(device), optimizer=np.random.choice(self.optimizers)), encoding



def cli(**kwargs):
    from tabular_logger import set_tlogger
    with open("architecture_search/architecture_search.json") as file:
        kwargs = dict(json.load(file), **kwargs)
    set_tlogger(kwargs.pop("name", "default"))
    return main(automl_class=AutoML, **kwargs)

if __name__ == "__main__":
    import fire
    fire.Fire(cli)
