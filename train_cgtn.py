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
from nas_bench_modules import NASBenchClassifier
import enas_micro_models as enas
import models
from tlogger import TLogger




def main(num_inner_iterations=64,
         noise_size=128,
         inner_loop_init_lr=0.2,
         inner_loop_init_momentum=0.5,
         training_iterations_schedule=5,
         min_training_iterations=4,
         lr=0.1,
         rms_momentum=0.9,
         final_relative_lr=1e-2,
         generator_batch_size=128,
         meta_batch_size=512,
         adam_epsilon=1e-8,
         adam_beta1=0.0,
         adam_beta2=0.999,
         num_meta_iterations=1000,
         starting_meta_iteration=1,
         max_elapsed_time=None,
         gradient_block_size=16,
         use_intermediate_losses=0,
         intermediate_losses_ratio=1.0,
         data_path='./data',
         meta_optimizer="adam",
         dataset='MNIST',
         logging_period=10,
         generator_type="cgtn",
         learner_type="base",
         validation_learner_type=None,
         warmup_iterations=None,
         warmup_learner="base",
         final_batch_norm_forward=False,
         # The following flag is used for architecture search (it maps iteration to a specific architecture)
         iteration_maps_seed=False,
         use_dataset_augmentation=False,
         training_schedule_backwards=True,
         evenly_distributed_labels=True,
         iterations_depth_schedule=100,
         use_encoder=True,
         decoder_loss_multiplier=1.0,
         load_from=None,
         virtual_batch_size=1,
         deterministic=False,
         seed=1,
         grad_bound=None,
         version=None,  # dummy variable
         enable_checkpointing=True,
         randomize_width=False,
         step_by_step_validation=True,
         semisupervised_classifier_loss=True,
         semisupervised_student_loss=True,

         inner_loop_optimizer="SGD",
         meta_learn_labels=False,
         device='cuda'):
    validation_learner_type = validation_learner_type or learner_type
    hvd.init()
    assert hvd.mpi_threads_supported()
    from mpi4py import MPI
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    lr = lr * virtual_batch_size * hvd.size()
    torch.cuda.set_device(hvd.local_rank())
    # Load dataset
    img_shape, trainset, validationset, (testset_x, testset_y) = get_dataset(
        dataset, data_path, seed, device, with_augmentation=use_dataset_augmentation)
    validation_x, validation_y = zip(*validationset)
    validation_x = torch.stack(validation_x).to(device)
    validation_y = torch.as_tensor(validation_y).to(device)
    # Make each worker slightly different
    torch.manual_seed(seed + hvd.rank())
    np.random.seed(seed + hvd.rank())

    if generator_type == "semisupervised":
        unlabelled_trainset, trainset = torch.utils.data.random_split(trainset, [49500, 500])

    data_loader = torch.utils.data.DataLoader(trainset, batch_size=meta_batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    data_loader = EndlessDataLoader(data_loader)

    if generator_type == "cgtn":
        generator = CGTN(
            generator=Generator(noise_size + 10, img_shape),
            num_inner_iterations=num_inner_iterations,
            generator_batch_size=generator_batch_size,
            noise_size=noise_size,
            evenly_distributed_labels=evenly_distributed_labels,
            meta_learn_labels=bool(meta_learn_labels),
        )
    elif generator_type == "cgtn_all_shuffled":
        generator = CGTNAllShuffled(
            generator=Generator(noise_size + 10, img_shape),
            num_inner_iterations=num_inner_iterations,
            generator_batch_size=generator_batch_size,
            noise_size=noise_size,
            evenly_distributed_labels=evenly_distributed_labels,
        )
    elif generator_type == "cgtn_batch_shuffled":
        generator = CGTNBatchShuffled(
            generator=Generator(noise_size + 10, img_shape),
            num_inner_iterations=num_inner_iterations,
            generator_batch_size=generator_batch_size,
            noise_size=noise_size,
            evenly_distributed_labels=evenly_distributed_labels,
        )
    elif generator_type == "gtn":
        generator = GTN(
            generator=Generator(noise_size + 10, img_shape),
            generator_batch_size=generator_batch_size,
            noise_size=noise_size,
        )
    elif generator_type == "gaussian_cgtn":
        generator = GaussianCGTN(
            generator=Generator(noise_size + 10, img_shape),
            num_inner_iterations=num_inner_iterations,
            generator_batch_size=generator_batch_size,
            noise_size=noise_size,
        )
    elif generator_type == "dataset":
        generator = UniformSamplingGenerator(
            torch.utils.data.DataLoader(trainset, batch_size=generator_batch_size, shuffle=True, drop_last=True),
            num_inner_iterations=num_inner_iterations,
            device=device,
        )
    elif generator_type == "distillation":
        generator = DatasetDistillation(
            img_shape=img_shape,
            num_inner_iterations=num_inner_iterations,
            generator_batch_size=generator_batch_size,
        )
    elif generator_type == "semisupervised":
        generator = SemisupervisedGenerator(
            torch.utils.data.DataLoader(unlabelled_trainset, batch_size=generator_batch_size, shuffle=True, drop_last=True),
            num_inner_iterations=num_inner_iterations,
            device=device,
            classifier=models.ClassifierLarger2(img_shape, batch_norm_momentum=0.9, randomize_width=False)
        )
    else:
        raise NotImplementedError()

    # Create meta-objective models
    if inner_loop_optimizer == "SGD":
        optimizers = [inner_optimizers.SGD(inner_loop_init_lr, inner_loop_init_momentum, num_inner_iterations)]
    elif inner_loop_optimizer == "RMSProp":
        optimizers = [inner_optimizers.RMSProp(inner_loop_init_lr, inner_loop_init_momentum, num_inner_iterations)]
    elif inner_loop_optimizer == "Adam":
        optimizers = [inner_optimizers.Adam(inner_loop_init_lr, inner_loop_init_momentum, num_inner_iterations)]
    else:
        raise ValueError(f"Inner loop optimizer '{inner_loop_optimizer}' not available")

    automl = AutoML(
        generator=generator,
        optimizers=optimizers,
    )
    if use_encoder:
        automl.encoder = Encoder(img_shape, output_size=noise_size)

    automl = automl.to(device)

    if meta_optimizer == "adam":
        optimizer = torch.optim.Adam(automl.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_epsilon)
    elif meta_optimizer == "sgd":
        optimizer = torch.optim.SGD(automl.parameters(), lr=lr, momentum=rms_momentum)
    elif meta_optimizer == "RMS":
        optimizer = torch.optim.RMSprop(automl.parameters(), lr=lr, alpha=adam_beta1, momentum=rms_momentum, eps=adam_epsilon)
    else:
        raise NotImplementedError()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_meta_iterations, lr * final_relative_lr)
    if hvd.rank() == 0:
        if load_from:
            state = torch.load(load_from)
            automl.load_state_dict(state["model"])
            if lr > 0:
                optimizer.load_state_dict(state["optimizer"])
            del state
            tlogger.info("loaded from:", load_from)
        total_num_parameters = 0
        for name, value in automl.named_parameters():
            tlogger.info("Optimizing parameter:", name, value.shape)
            total_num_parameters += np.prod(value.shape)
        tlogger.info("Total number of parameters:", int(total_num_parameters))

    def compute_learner(learner, iterations=num_inner_iterations, keep_grad=True, callback=None):
        learner.model.train()
        names, params = list(zip(*learner.model.get_parameters()))
        buffers = list(zip(*learner.model.named_buffers()))
        if buffers:
            buffer_names, buffers = buffers
        else:
            buffer_names, buffers = None, ()
        param_shapes = [p.shape for p in params]
        param_sizes = [np.prod(shape) for shape in param_shapes]
        param_end_point = np.cumsum(param_sizes)

        buffer_shapes = [p.shape for p in buffers]
        buffer_sizes = [np.prod(shape) for shape in buffer_shapes]
        buffer_end_point = np.cumsum(buffer_sizes)

        def split_params(fused_params):
            # return fused_params
            assert len(fused_params) == 1
            return [fused_params[0][end - size:end].reshape(shape) for end, size, shape in zip(param_end_point, param_sizes, param_shapes)]

        def split_buffer(fused_params):
            if fused_params:
                # return fused_params
                assert len(fused_params) == 1
                return [fused_params[0][end - size:end].reshape(shape) for end, size, shape in zip(buffer_end_point, buffer_sizes, buffer_shapes)]
            return fused_params

        # test = split_params(torch.cat([p.flatten() for p in params]))
        # assert all([np.allclose(params[i].detach().cpu(), test[i].detach().cpu()) for i in range(len(test))])
        params = [torch.cat([p.flatten() for p in params])]
        buffers = [torch.cat([p.flatten() for p in buffers])] if buffers else buffers
        optimizer_state = learner.optimizer.initial_state(params)

        params = params, buffers
        initial_params = nest.map_structure(lambda p: None, params)

        losses = {}
        accuracies = {}

        def intermediate_loss(params):
            params = nest.pack_sequence_as(initial_params, params[1:])
            params, buffers = params
            x, y = next(meta_generator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            learner.model.set_parameters(list(zip(names, split_params(params))))
            if buffer_names:
                learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))
            learner.model.eval()
            pred = learner.model(x)
            if isinstance(pred, tuple):
                pred, aux_pred = pred
                loss = F.nll_loss(pred, y) + F.nll_loss(aux_pred, y)
            else:
                loss = F.nll_loss(pred, y)
            return loss * intermediate_losses_ratio

        if hasattr(automl.generator, "init"):
            generator_args = [automl.generator.init()]
        else:
            generator_args = []

        def body(args):
            it, params, optimizer_state = args
            if training_schedule_backwards:
                x, y_one_hot = automl.generator(iterations - it - 1, *generator_args)
            else:
                x, y_one_hot = automl.generator(it, *generator_args)
            with torch.enable_grad():
                if use_intermediate_losses > 0 and (it >= use_intermediate_losses and it % use_intermediate_losses == 0):
                    params = SurrogateLoss.apply(intermediate_loss, it, *nest.flatten(params))
                    params = nest.pack_sequence_as(initial_params, params[1:])
                params, buffers = params
                for p in params:
                    if not p.requires_grad:
                        p.requires_grad = True

                learner.model.set_parameters(list(zip(names, split_params(params))))
                if buffer_names:
                    learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))
                learner.model.train()
                output = learner.model(x)
                if isinstance(output, tuple):
                    output1, output2 = output
                    loss = - (output1 * y_one_hot).sum() * (1 / output1.shape[0])
                    loss = loss - (output2 * y_one_hot).sum() * (1 / output2.shape[0])
                    pred = output1
                else:
                    loss = -(output * y_one_hot).sum() * (1 / output.shape[0])
                    pred = output
                if it.item() not in losses:
                    losses[it.item()] = loss.detach().cpu().item()
                    accuracies[it.item()] = (pred.max(-1).indices == y_one_hot.max(-1).indices).to(torch.float).mean().item()

                grads = grad(loss, params, create_graph=x.requires_grad, allow_unused=True)
            # assert len(grads) == len(names)
            new_params, optimizer_state = learner.optimizer(it, params, grads, optimizer_state)
            buffers = list(learner.model.buffers())
            buffers = [torch.cat([b.flatten() for b in buffers])] if buffers else buffers
            if callback is not None:
                learner.model.set_parameters(list(zip(names, split_params(params))))
                if buffer_names:
                    learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))
                callback(learner)

            return (it + 1, (new_params, buffers,), optimizer_state)

        last_state, params, optimizer_state = gradient_checkpointing((torch.as_tensor(0), params, optimizer_state), body, iterations,
                                                                     block_size=gradient_block_size)
        assert last_state.item() == iterations
        params, buffers = params
        learner.model.set_parameters(list(zip(names, split_params(params))))
        if buffer_names:
            learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))

        if final_batch_norm_forward:
            x, _ = automl.generator(torch.randint(iterations, size=()))
            learner.model.train()
            learner.model(x)

        return learner, losses, accuracies

    tstart = time.time()
    meta_generator = iter(data_loader)
    hvd.broadcast_parameters(automl.state_dict(), root_rank=0)
    best_optimizers = {}
    validation_accuracy = None
    total_inner_iterations_so_far = 0
    for iteration in range(starting_meta_iteration, num_meta_iterations + 1):
        last_iteration = time.time()
        # basic logging
        tlogger.record_tabular('Iteration', iteration)
        tlogger.record_tabular('lr', optimizer.param_groups[0]['lr'])

        # Train learner
        if training_iterations_schedule > 0:
            training_iterations = int(min(num_inner_iterations, min_training_iterations +
                                          (iteration - starting_meta_iteration) // training_iterations_schedule))
        else:
            training_iterations = num_inner_iterations
        tlogger.record_tabular('training_iterations', training_iterations)
        total_inner_iterations_so_far += training_iterations
        tlogger.record_tabular('training_iterations_so_far', total_inner_iterations_so_far * hvd.size())

        optimizer.zero_grad()

        for _ in range(virtual_batch_size):
            torch.cuda.empty_cache()
            meta_x, meta_y = next(meta_generator)
            meta_x = meta_x.to('cuda', non_blocking=True)
            meta_y = meta_y.to('cuda', non_blocking=True)

            tstart_forward = time.time()
            if generator_type != "semisupervised" or semisupervised_student_loss:
                # TODO: Learn batchnorm momentum and eps
                sample_learner_type = learner_type
                if warmup_iterations is not None and iteration < warmup_iterations:
                    sample_learner_type = warmup_learner
                learner, encoding = automl.sample_learner(img_shape, device,
                                                          allow_nas=False,
                                                          randomize_width=randomize_width,
                                                          learner_type=sample_learner_type,
                                                          iteration_maps_seed=iteration_maps_seed,
                                                          iteration=iteration,
                                                          deterministic=deterministic,
                                                          iterations_depth_schedule=iterations_depth_schedule)
                automl.train()

                if lr == 0.0:
                    with torch.no_grad():
                        learner, intermediate_losses, intermediate_accuracies = compute_learner(learner, iterations=training_iterations, keep_grad=lr > 0.0)
                else:
                    learner, intermediate_losses, intermediate_accuracies = compute_learner(learner, iterations=training_iterations, keep_grad=lr > 0.0)
                # TODO: remove this requirement
                params = list(learner.model.get_parameters())
                learner.model.eval()

                # Evaluate learner on training and back-prop
                torch.cuda.empty_cache()
                pred = learner.model(meta_x)
                if isinstance(pred, tuple):
                    pred, aux_pred = pred
                    loss = F.nll_loss(pred, meta_y) + F.nll_loss(aux_pred, meta_y)
                else:
                    loss = F.nll_loss(pred, meta_y)
                accuracy = (pred.max(-1).indices == meta_y).to(torch.float).mean()
                tlogger.record_tabular("TimeElapsedForward", time.time() - tstart_forward)
                num_parameters = sum([a[1].size().numel() for a in params])
                tlogger.record_tabular("TrainingLearnerParameters", num_parameters)
                tlogger.record_tabular("optimizer", type(learner.optimizer).__name__)
                tlogger.record_tabular('meta_training_loss', loss.item())
                tlogger.record_tabular('meta_training_accuracy', accuracy.item())
                tlogger.record_tabular('training_accuracies', intermediate_accuracies)
                tlogger.record_tabular('training_losses', intermediate_losses)
                tlogger.record_tabular("dag", encoding)
            else:
                loss = torch.as_tensor(0.0)

            if lr > 0.0:
                tstart_backward = time.time()
                if generator_type != "semisupervised" or semisupervised_student_loss:
                    loss.backward()

                if generator_type == "semisupervised" and semisupervised_classifier_loss:
                    automl.generator.classifier.train()
                    pred = automl.generator.classifier(meta_x)
                    accuracy = (pred.max(-1).indices == meta_y).to(torch.float).mean()
                    loss2 = F.nll_loss(pred, meta_y)
                    loss2.backward()
                    tlogger.record_tabular('meta_training_generator_loss', loss2.item())
                    tlogger.record_tabular('meta_training_generator_accuracy', accuracy.item())
                    loss = loss + loss2
                    del loss2

                tlogger.record_tabular("TimeElapsedBackward", time.time() - tstart_backward)

                if use_encoder:
                    # TODO: add loss weight
                    meta_encoding = automl.encoder(meta_x)

                    meta_y_one_hot = torch.zeros(meta_x.shape[0], 10, device=device)
                    meta_y_one_hot.scatter_(1, meta_y.unsqueeze(-1), 1)
                    meta_encoding = torch.cat([meta_encoding, meta_y_one_hot], -1)
                    reconstruct = automl.generator.generator(meta_encoding)
                    ae_loss = decoder_loss_multiplier * F.mse_loss(reconstruct, meta_x)
                    ae_loss.backward()
                    tlogger.record_tabular("decoder_loss", ae_loss.item())

        if lr > 0.0:
            # If using distributed training aggregard gradients with Horovod
            maybe_allreduce_grads(automl)
            if grad_bound is not None:
                nn.utils.clip_grad_norm_(automl.parameters(), grad_bound)
            optimizer.step()
            if max_elapsed_time is not None:
                scheduler.step(round((time.time() - tstart) / max_elapsed_time * num_meta_iterations))
            else:
                scheduler.step(iteration - 1)

        is_last_iteration = iteration == num_meta_iterations or (max_elapsed_time is not None and time.time() - tstart > max_elapsed_time)
        if np.isnan(loss.item()):
            tlogger.info("NaN training loss, terminating")
            is_last_iteration = True
        is_last_iteration = MPI.COMM_WORLD.bcast(is_last_iteration, root=0)
        if iteration == 1 or iteration % logging_period == 0 or is_last_iteration:
            tstart_validation = time.time()

            val_loss, val_accuracy = [], []
            test_loss, test_accuracy = [], []

            if generator_type == "semisupervised":
                # Validation set
                evaluate_set(generator.classifier, validation_x, validation_y, "generator_validation")
                # Test set
                evaluate_set(generator.classifier, testset_x, testset_y, "generator_test")
            else:
                def compute_learner_callback(learner):
                    # Validation set
                    validation_loss, single_validation_accuracy, validation_accuracy = evaluate_set(learner.model, validation_x, validation_y, "validation")
                    val_loss.append(validation_loss)
                    val_accuracy.append(validation_accuracy)
                    best_optimizers[type(learner.optimizer).__name__] = single_validation_accuracy.item()
                    # Test set
                    loss, _, accuracy = evaluate_set(learner.model, testset_x, testset_y, "test")
                    test_loss.append(loss)
                    test_accuracy.append(accuracy)

                tlogger.info("sampling another learner_type ({}) for validation".format(validation_learner_type))
                learner, _ = automl.sample_learner(img_shape, device,
                                                   allow_nas=False,
                                                   learner_type=validation_learner_type,
                                                   iteration_maps_seed=iteration_maps_seed,
                                                   iteration=iteration,
                                                   deterministic=deterministic,
                                                   iterations_depth_schedule=iterations_depth_schedule
                                                   )
                if step_by_step_validation:
                    compute_learner_callback(learner)
                with torch.no_grad():
                    learner, _, _ = compute_learner(learner, iterations=training_iterations, keep_grad=False,
                                                    callback=compute_learner_callback if step_by_step_validation else None)
                if not step_by_step_validation:
                    compute_learner_callback(learner)

                tlogger.record_tabular("validation_losses", val_loss)
                tlogger.record_tabular("validation_accuracies", val_accuracy)
                validation_accuracy = val_accuracy[-1]
                tlogger.record_tabular("test_losses", test_loss)
                tlogger.record_tabular("test_accuracies", test_accuracy)

            # Extra logging
            tlogger.record_tabular('TimeElapsedIter', (tstart_validation - last_iteration) / virtual_batch_size)
            tlogger.record_tabular('TimeElapsedValidation', time.time() - tstart_validation)
            tlogger.record_tabular('TimeElapsed', time.time() - tstart)

            for k, v in best_optimizers.items():
                tlogger.record_tabular("{}_last_accuracy".format(k), v)

            if hvd.rank() == 0:
                tlogger.dump_tabular()

                if (iteration == 1 or iteration % 1000 == 0 or is_last_iteration):
                    with torch.no_grad():
                        if enable_checkpointing:
                            batches = []
                            for it in range(num_inner_iterations):
                                if training_schedule_backwards:
                                    x, y = automl.generator(num_inner_iterations - it - 1)
                                else:
                                    x, y = automl.generator(it)
                                batches.append((x.cpu().numpy(), y.cpu().numpy()))
                            batches = list(reversed(batches))
                            with open(os.path.join(tlogger.get_dir(), 'samples_{}.pkl'.format(iteration)), 'wb') as file:
                                pickle.dump(batches, file)
                            del batches
                            tlogger.info("Saved:", os.path.join(tlogger.get_dir(), 'samples_{}.pkl'.format(iteration)))
                        ckpt = os.path.join(tlogger.get_dir(), 'checkpoint_{}.pkl'.format(iteration))
                        torch.save({
                            "optimizer": optimizer.state_dict(),
                            "model": automl.state_dict(),
                        }, ckpt)
                        tlogger.info("Saved:", ckpt)

            if is_last_iteration:
                break
        elif hvd.rank() == 0:
            tlogger.info("training_loss:", loss.item())
    return validation_accuracy


class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_dataset(dataset, data_path, seed, device, with_augmentation=False, cutout_size=16):
    torch.manual_seed(seed)
    # Load dataset
    if dataset == "MNIST":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = datasets.MNIST(root=data_path, train=True, download=hvd.rank() == 0, transform=transform)
        trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
        testset = datasets.MNIST(root=data_path, train=False, download=hvd.rank() == 0, transform=transform)
        img_shape = (1, 28, 28)
    elif dataset == "CIFAR10":
        img_mean = [0.49139968, 0.48215827, 0.44653124]
        img_std = [0.24703233, 0.24348505, 0.26158768]
        # img_mean = [0.5, 0.5, 0.5]
        # img_std = [0.5, 0.5, 0.5]
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(img_mean, img_std),
        ])
        if with_augmentation:
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(img_mean, img_std),
                # Cutout(cutout_size),
            ])
        else:
            transform_train = transform
        img_mean = torch.as_tensor(img_mean).to(device)[..., None, None]
        img_std = torch.as_tensor(img_std).to(device)[..., None, None]
        trainset = datasets.CIFAR10(root=data_path, train=True, download=hvd.rank() == 0, transform=transform_train)
        validationset = datasets.CIFAR10(root=data_path, train=True, download=hvd.rank() == 0, transform=transform)

        # Split train and validationset
        lengths = [45000, 5000]
        indices = torch.randperm(sum(lengths)).tolist()
        trainset = torch.utils.data.dataset.Subset(trainset, indices[:lengths[0]])
        validationset = torch.utils.data.dataset.Subset(validationset, indices[-lengths[1]:])

        testset = datasets.CIFAR10(root=data_path, train=False, download=hvd.rank() == 0, transform=transform)
        img_shape = (3, 32, 32)
    else:
        raise ValueError("Invalid dataset {}".format(dataset))
    # Make each worker slightly different
    torch.manual_seed(seed + hvd.rank())

    # validation
    # TODO: Remove augmentation or make it "dynamic"
    testset_x, testset_y = zip(*testset)
    testset_x = torch.stack(testset_x).to(device)
    testset_y = torch.as_tensor(testset_y).to(device)

    return img_shape, trainset, validationset, (testset_x, testset_y)


def maybe_allreduce_grads(model):
    if hvd.size() > 1:
        tstart_reduce = time.time()
        named_parameters = list(sorted(model.named_parameters(), key=lambda a: a[0]))
        grad_handles = []
        for name, p in named_parameters:
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                with torch.no_grad():
                    grad_handles.append(hvd.allreduce_async_(p.grad, name=name))
        for handle in grad_handles:
            hvd.synchronize(handle)
        tlogger.record_tabular("TimeElapsedAllReduce", time.time() - tstart_reduce)
        if time.time() - tstart_reduce > 5:
            import socket
            tlogger.info("Allreduce took more than 5 seconds for node {} (rank {})".format(socket.gethostname(), hvd.rank()))


def evaluate_set(model, x, y, name):
    with torch.no_grad():
        batch_size = 1000
        validation_pred = []
        model.eval()
        for i in range(math.ceil(len(x) / batch_size)):
            pred = model(x[i * batch_size:(i + 1) * batch_size])
            if isinstance(pred, tuple):
                pred, _ = pred
            validation_pred.append(pred)
        validation_pred = torch.cat(validation_pred, dim=0)
        single_validation_accuracy = (validation_pred.max(-1).indices == y).to(torch.float).mean()

        ensemble_pred = hvd.allreduce(torch.exp(validation_pred), name="{}_ensemble_pred".format(name))
        ensemble_validation_accuracy = (ensemble_pred.max(-1).indices == y).to(torch.float).mean()

        validation_accuracy = hvd.allreduce(single_validation_accuracy, name="{}_accuracy".format(name)).item()
        validation_loss = hvd.allreduce(F.nll_loss(validation_pred, y), name="{}_loss".format(name)).item()
        tlogger.record_tabular('{}_loss'.format(name), validation_loss)
        tlogger.record_tabular('{}_accuracy'.format(name), validation_accuracy)
        tlogger.record_tabular('ensemble_{}_accuracy'.format(name), ensemble_validation_accuracy)
        return validation_loss, single_validation_accuracy, validation_accuracy


class CGTN(nn.Module):
    def __init__(self, generator, num_inner_iterations, generator_batch_size, noise_size, evenly_distributed_labels=False, meta_learn_labels=False):
        super().__init__()
        self.generator_batch_size = generator_batch_size
        self.generator = generator
        if evenly_distributed_labels:
            labels = torch.arange(num_inner_iterations * generator_batch_size) % 10
            labels = torch.reshape(labels, (num_inner_iterations, generator_batch_size))
            self.curriculum_labels = nn.Parameter(labels, requires_grad=False)
        else:
            self.curriculum_labels = nn.Parameter(torch.randint(10, size=(num_inner_iterations, generator_batch_size), dtype=torch.int64), requires_grad=False)
        self.curriculum_labels_one_hot = torch.zeros(num_inner_iterations, generator_batch_size, 10)
        self.curriculum_labels_one_hot.scatter_(2, self.curriculum_labels.unsqueeze(-1), 1)
        self.curriculum_labels_one_hot = nn.Parameter(self.curriculum_labels_one_hot, requires_grad=meta_learn_labels)
        # TODO: Maybe learn the soft-labels?
        self.curriculum = nn.Parameter(torch.randn((num_inner_iterations, generator_batch_size, noise_size), dtype=torch.float32))
        self.generator = torch.jit.trace(self.generator, (torch.rand(generator_batch_size, noise_size + 10),))

    def forward(self, it):
        label = self.curriculum_labels_one_hot[it]
        noise = torch.cat([self.curriculum[it], label], dim=-1)
        x = self.generator(noise)
        if not x.requires_grad:
            label = label.detach()
        return x, label


class CGTNAllShuffled(CGTN):
    def forward(self, it):
        all_images = torch.reshape(self.curriculum, (-1,) + self.curriculum.shape[2:])
        all_labels = torch.reshape(self.curriculum_labels_one_hot, (-1,) + self.curriculum_labels_one_hot.shape[2:])
        idx = torch.randint(len(all_images), size=(self.generator_batch_size,), device=all_images.device)
        noise = all_images[idx]
        labels = all_labels[idx]
        noise = torch.cat([noise, labels], dim=-1)
        x = self.generator(noise)
        return x, labels


class CGTNBatchShuffled(CGTN):
    def forward(self, it):
        idx = torch.randint(len(self.curriculum), size=())
        noise = self.curriculum[idx]
        labels = self.curriculum_labels_one_hot[idx]
        noise = torch.cat([noise, labels], dim=-1)
        x = self.generator(noise)
        return x, labels


class GTN(nn.Module):
    def __init__(self, generator, generator_batch_size, noise_size):
        super().__init__()
        self.generator = generator
        self.generator_batch_size = generator_batch_size
        self.noise_size = noise_size
        self.generator = torch.jit.trace(self.generator, (torch.rand(generator_batch_size, noise_size + 10),))

    def forward(self, it):
        curriculum_labels = torch.randint(10, size=(self.generator_batch_size,), dtype=torch.int64, device="cuda")
        curriculum_labels_one_hot = torch.zeros(self.generator_batch_size, 10, device="cuda")
        curriculum_labels_one_hot.scatter_(1, curriculum_labels.unsqueeze(-1), 1)
        curriculum_labels_one_hot = curriculum_labels_one_hot.to("cuda")

        noise = torch.cat([torch.randn(self.generator_batch_size, self.noise_size, device="cuda"), curriculum_labels_one_hot], dim=-1)
        x = self.generator(noise)
        return x, curriculum_labels_one_hot


class DatasetDistillation(nn.Module):
    def __init__(self, num_inner_iterations, generator_batch_size, img_shape):
        super().__init__()
        self.curriculum_labels = nn.Parameter(torch.randint(10, size=(num_inner_iterations, generator_batch_size), dtype=torch.int64), requires_grad=False)
        self.curriculum_labels_one_hot = torch.zeros(num_inner_iterations, generator_batch_size, 10)
        self.curriculum_labels_one_hot.scatter_(2, self.curriculum_labels.unsqueeze(-1), 1)
        self.curriculum_labels_one_hot = nn.Parameter(self.curriculum_labels_one_hot, requires_grad=False)
        self.curriculum = nn.Parameter(torch.randn((num_inner_iterations, generator_batch_size,) + img_shape, dtype=torch.float32))

    def forward(self, it):
        x = self.curriculum[it]
        return torch.tanh(x) * 2, self.curriculum_labels_one_hot[it]


class GaussianCGTN(nn.Module):
    def __init__(self, generator, num_inner_iterations, generator_batch_size, noise_size):
        super().__init__()
        self.generator = generator
        self.curriculum_labels = nn.Parameter(torch.randint(10, size=(num_inner_iterations, generator_batch_size), dtype=torch.int64), requires_grad=False)
        self.curriculum_labels_one_hot = torch.zeros(num_inner_iterations, generator_batch_size, 10)
        self.curriculum_labels_one_hot.scatter_(2, self.curriculum_labels.unsqueeze(-1), 1)
        self.curriculum_labels_one_hot = nn.Parameter(self.curriculum_labels_one_hot, requires_grad=False)
        self.curriculum_mu = nn.Parameter(torch.zeros((num_inner_iterations, generator_batch_size, noise_size), dtype=torch.float32))
        self.curriculum_log_sigma = nn.Parameter(torch.ones((num_inner_iterations, generator_batch_size, noise_size), dtype=torch.float32))

    def forward(self, it):
        noise = self.curriculum_mu[it] + torch.exp(self.curriculum_log_sigma[it]) * torch.randn_like(self.curriculum_log_sigma[it])
        noise = torch.cat([noise, self.curriculum_labels_one_hot[it]], dim=-1)
        x = self.generator(noise)
        return torch.tanh(x) * 2, self.curriculum_labels_one_hot[it]


class UniformSamplingGenerator(nn.Module):
    def __init__(self, dataset, num_inner_iterations, device):
        super().__init__()
        self.sampler = iter(EndlessDataLoader(dataset))
        self.num_inner_iterations = num_inner_iterations
        self.device = device

    def init(self):
        batches = []
        for _ in range(self.num_inner_iterations):
            x, y = next(self.sampler)
            labels_one_hot = torch.zeros(y.size(0), 10)
            labels_one_hot.scatter_(1, y.unsqueeze(-1), 1)
            batches.append((nn.Parameter(x), nn.Parameter(labels_one_hot)))
        return batches

    def forward(self, x, batches=None):
        if batches is None:
            x, y = next(self.sampler)
            labels_one_hot = torch.zeros(y.size(0), 10)
            labels_one_hot.scatter_(1, y.unsqueeze(-1), 1)
            batches = [x, labels_one_hot]
            x = 0
        return batches[x][0].to(self.device), batches[x][1].to(self.device)


class SemisupervisedGenerator(UniformSamplingGenerator):
    def __init__(self, *args, classifier, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = classifier

    def forward(self, *args):
        x, _ = super().forward(*args)
        t = torch.exp(self.classifier(x))
        return x, t


class Learner(nn.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer


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

        if allow_nas and torch.randint(10, ()) == 0:
            model = NASBenchClassifier(input_shape, matrix, vertices, num_stacks=1, num_modules_per_stack=1)
        elif learner_type == "enas":
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


class EndlessDataLoader(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def __iter__(self):
        while True:
            for batch in self._data_loader:
                yield batch

def cli(**kwargs):
    from tabular_logger import TLogger
    global tlogger
    with open("experiments/cgtn.json") as file:
        kwargs = dict(json.load(file), **kwargs)
        tlogger = TLogger(kwargs.pop("name", "default"))
    return main(**kwargs)

if __name__ == "__main__":
    import fire
    fire.Fire(cli)
