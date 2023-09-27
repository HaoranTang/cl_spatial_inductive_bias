import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torchvision.models import resnet18, resnet50
# plotting
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchmetrics.classification import CalibrationError, Accuracy

from solo.args.setup import parse_args_linear
from solo.methods.base import BaseMethod
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)

DATASTATS = {"cifar10": 10, # num_cls
             "cifar100": 100,
             "mnist": 10,}

try:
    from solo.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
import types

from solo.methods.linear import LinearModel
# from solo.methods.sup import Sup
from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data


def main():
    args = parse_args_linear()

    assert args.backbone in BaseMethod._BACKBONES
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
    }[args.backbone]

    # initialize backbone
    kwargs = args.backbone_args
    cifar = kwargs.pop("cifar", False)
    # swin specific
    if "swin" in args.backbone and cifar:
        kwargs["window_size"] = 4

    backbone = backbone_model(**kwargs)
    if "resnet" in args.backbone:
        # remove fc layer
        backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            raise Exception(
                "You are using an older checkpoint."
                "Either use a new one, or convert it by replacing"
                "all 'encoder' occurances in state_dict with 'backbone'"
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)

    print(f"loaded {ckpt_path}")

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        Class = types.new_class(f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel))
    else:
        Class = LinearModel

    
    del args.backbone

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        corrupt=args.corrupt,
    )
    if args.dataset == "imagenetcls":
        args.num_classes = val_loader.dataset.num_classes
    
    model = Class(backbone, **args.__dict__)
    ece = CalibrationError(task='multiclass', n_bins=15, norm='l1', num_classes=DATASTATS[args.dataset])
    accuracy = Accuracy(task="multiclass", num_classes=DATASTATS[args.dataset])

    print(backbone)

    ### Uniformity etc before fine-tuning
    feats = []
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images.cuda()
            labels.cuda()
            full_out = model(images)
            pred = full_out['logits']
            feats.append(F.normalize(full_out['feats']))
            preds.append(pred)
            targets.append(labels)
    
    feats = torch.cat(feats)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    print(preds.shape, targets.shape)

    total_uniform = uniform_loss(feats)
    class_uniform = []
    for i in range(10):
        cls_feats = feats[targets==i]
        class_uniform.append(uniform_loss(cls_feats))
    print('Before finetuning stats...')
    print('Accuracy', accuracy(F.softmax(preds, dim=1), targets), 'total uniformity: ', total_uniform, " Exp Calibration Err: ", ece(preds, targets))
    print(class_uniform)
    
    # Plot those points as a scatter plot and label them based on the pred labels

    # feats = feats.detach().cpu()
    # targets = targets.cpu()
    # tsne = TSNE(2, verbose=0, random_state=0)
    # tsne_proj = tsne.fit_transform(feats)
    
    # cmap = cm.get_cmap('tab20')
    # fig, ax = plt.subplots(figsize=(8,8))
    # num_categories = 10
    # for lab in range(num_categories):
    #     indices = targets==lab
    #     ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    # ax.legend(fontsize='large', markerscale=2)
    # plt.title('SL Original (uniformity={0:.2f})'.format(total_uniform), fontsize=25)
    # plt.savefig('sup_unif_ori.svg', format='svg')

    callbacks = []
    if args.wandb or args.csv:
        if args.wandb:
            logger = WandbLogger(
                name=args.name,
                project=args.project,
                entity=args.entity,
                offline=args.offline,
            )
            logger.watch(model, log="gradients", log_freq=100)
        else:
            logger = CSVLogger(
                save_dir='./csv',
                name=args.name,
                version=args.corrupt,
            )
        logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            # logdir=os.path.join(args.checkpoint_dir, "linear"),
            logdir=os.path.join(args.checkpoint_dir, args.name),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint
    else:
        ckpt_path = None

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger if args.wandb or args.csv else None,
        callbacks=callbacks,
        enable_checkpointing=False,
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    ### Uniformity etc after fine-tuning
    feats = []
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images.cuda()
            labels.cuda()
            full_out = model(images)
            pred = full_out['logits']
            feats.append(F.normalize(full_out['feats']))
            preds.append(pred)
            targets.append(labels)
    
    feats = torch.cat(feats)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    print(preds.shape, targets.shape)

    total_uniform = uniform_loss(feats)
    class_uniform = []
    for i in range(10):
        cls_feats = feats[targets==i]
        class_uniform.append(uniform_loss(cls_feats))
    print('After finetuning stats...')
    print('Accuracy', accuracy(F.softmax(preds, dim=1), targets), 'total uniformity: ', total_uniform, " Exp Calibration Err: ", ece(preds, targets))
    print(class_uniform)

def uniform_loss(x, t=2):
    # return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    return -torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


if __name__ == "__main__":
    main()
