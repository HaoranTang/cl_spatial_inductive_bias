# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.methods.base import BaseMethod
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.misc import compute_dataset_size
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torchvision.transforms import CenterCrop


class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        lr_decay_steps: Optional[Sequence[int]] = None,
        no_channel_last: bool = False,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            weight_decay (float): weight decay.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
            no_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
        """

        super().__init__()

        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features
        self.classifier = nn.Linear(features_dim, num_classes)  # type: ignore

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps
        self.no_channel_last = no_channel_last

        self._num_training_steps = None

        # all the other parameters
        self.extra_args = kwargs

        for param in self.backbone.parameters():
            param.requires_grad = False

        # can provide up to ~20% speed up
        if not no_channel_last:
            self = self.to(memory_format=torch.channels_last)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")

        # backbone args
        parser.add_argument("--backbone", choices=BaseMethod._BACKBONES, type=str)
        # for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        parser.add_argument(
            "--optimizer", choices=LinearModel._OPTIMIZERS.keys(), type=str, required=True
        )
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        parser.add_argument(
            "--scheduler", choices=LinearModel._SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        return parent_parser

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if self._num_training_steps is None:
            try:
                dataset = self.extra_args.get("dataset", None)
                if dataset not in ["cifar10", "cifar100", "stl10"]:
                    data_dir = self.extra_args.get("data_dir", ".")
                    train_dir = self.extra_args.get("train_dir", "train")
                    folder = os.path.join(data_dir, train_dir)
                else:
                    folder = None
                no_labels = self.extra_args.get("no_labels", False)
                data_fraction = self.extra_args.get("data_fraction", -1.0)

                dataset_size = compute_dataset_size(
                    dataset=dataset,
                    folder=folder,
                    train=True,
                    no_labels=no_labels,
                    data_fraction=data_fraction,
                )
            except:
                raise RuntimeError(
                    "Please pass 'dataset' or 'data_dir '"
                    "and 'train_dir' as parameters to the model."
                )

            dataset_size = self.trainer.limit_train_batches * dataset_size

            num_devices = self.trainer.num_devices
            num_nodes = self.extra_args.get("num_nodes_horovod", 0) or self.trainer.num_nodes or 1
            effective_batch_size = (
                self.batch_size * self.trainer.accumulate_grad_batches * num_devices * num_nodes
            )
            self._num_training_steps = dataset_size // effective_batch_size

        return self._num_training_steps

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        with torch.no_grad():
            feats = self.backbone(X)
        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs * self.num_training_steps,
                    max_epochs=self.max_epochs * self.num_training_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def shared_step(
        self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        X, target = batch
        batch_size = X.size(0)

        out = self(X)["logits"]

        loss = F.cross_entropy(out, target)

        acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
        return batch_size, loss, acc1, acc5

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        self.backbone.eval()

        _, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        batch_size, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)

class SegmDecoder(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        lr_decay_steps: Optional[Sequence[int]] = None,
        no_channel_last: bool = False,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            weight_decay (float): weight decay.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
            no_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
        """

        super().__init__()

        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features
        self.decoder = decoder_segm(features_dim, num_classes)  # type: ignore

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps
        self.no_channel_last = no_channel_last

        self._num_training_steps = None

        # all the other parameters
        self.extra_args = kwargs

        for param in self.backbone.parameters():
            param.requires_grad = False

        # can provide up to ~20% speed up
        if not no_channel_last:
            self = self.to(memory_format=torch.channels_last)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")

        # backbone args
        parser.add_argument("--backbone", choices=BaseMethod._BACKBONES, type=str)
        # for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        parser.add_argument(
            "--optimizer", choices=SegmDecoder._OPTIMIZERS.keys(), type=str, required=True
        )
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        parser.add_argument(
            "--scheduler", choices=SegmDecoder._SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        return parent_parser

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if self._num_training_steps is None:
            try:
                dataset = self.extra_args.get("dataset", None)
                if dataset not in ["cifar10", "cifar100", "stl10"]:
                    data_dir = self.extra_args.get("data_dir", ".")
                    train_dir = self.extra_args.get("train_dir", "train")
                    folder = os.path.join(data_dir, train_dir)
                else:
                    folder = None
                no_labels = self.extra_args.get("no_labels", False)
                data_fraction = self.extra_args.get("data_fraction", -1.0)

                dataset_size = compute_dataset_size(
                    dataset=dataset,
                    folder=folder,
                    train=True,
                    no_labels=no_labels,
                    data_fraction=data_fraction,
                )
            except:
                raise RuntimeError(
                    "Please pass 'dataset' or 'data_dir '"
                    "and 'train_dir' as parameters to the model."
                )

            dataset_size = self.trainer.limit_train_batches * dataset_size

            num_devices = self.trainer.num_devices
            num_nodes = self.extra_args.get("num_nodes_horovod", 0) or self.trainer.num_nodes or 1
            effective_batch_size = (
                self.batch_size * self.trainer.accumulate_grad_batches * num_devices * num_nodes
            )
            self._num_training_steps = dataset_size // effective_batch_size

        return self._num_training_steps

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        with torch.no_grad():
            feats = self.backbone(X)
        logits = self.decoder(feats)
        return {"logits": logits, "feats": feats}

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            self.decoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs * self.num_training_steps,
                    max_epochs=self.max_epochs * self.num_training_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def shared_step(
        self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        X, target = batch
        batch_size = X.size(0)

        out = self(X)["logits"]
        print('SHAPES, ', out.shape, target.shape)
        loss = F.cross_entropy(out, target)

        acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
        return batch_size, loss, acc1, acc5

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        self.backbone.eval()

        _, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        batch_size, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c, out_c)
    def forward(self, inputs):
        x = self.up(inputs)
        x = self.conv(x)
        return x
    
# class decoder_segm(nn.Module):
#     def __init__(self, features_dim, num_classes):
#         super().__init__()
#         """ Decoder """
#         self.d1 = decoder_block(features_dim, 1024)
#         self.d2 = decoder_block(1024, 512)
#         self.d3 = decoder_block(512, 256)
#         self.d4 = decoder_block(256, 128)
#         self.d5 = decoder_block(128, 64)
#         """ Classifier """
#         self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
#     def forward(self, feats):
#         print(feats.shape)
#         """ Decoder """
#         d1 = self.d1(feats[:,:,None, None])
#         d2 = self.d2(d1)
#         d3 = self.d3(d2)
#         d4 = self.d4(d3)
#         d5 = self.d5(d4)
#         """ Classifier """
#         outputs = self.outputs(d5)
#         return outputs

class decoder_segm(nn.Module):
    def __init__(self, input_dim, output_channels):
        super(decoder_segm, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(input_dim, 512, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.conv = nn.Conv2d(32, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, features):
        x = features.view(features.size(0), -1, 1, 1)
        print(features.shape, x.shape)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.relu(x)
        x = self.deconv5(x)
        x = self.relu(x)
        x = self.conv(x)
        print(x.shape)

        return x

class Block(nn.Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = nn.Conv2d(inChannels, outChannels, 3)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))

class Decoder(nn.Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = nn.ModuleList(
			[nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = nn.ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures