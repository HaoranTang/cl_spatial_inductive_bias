import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.methods.base_old import BaseMethod


class Sup(BaseMethod):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # projector
        # self.projector = nn.Linear(self.features_dim, self.num_classes)
        self.backbone_detach = False

    # @property
    # def learnable_params(self) -> List[dict]:
    #     """Adds projector parameters to the parent's learnable parameters.

    #     Returns:
    #         List[dict]: list of learnable parameters.
    #     """

    #     extra_learnable_params = [{"params": self.projector.parameters()}]
    #     return super().learnable_params + extra_learnable_params

    # def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
    #     """Performs the forward pass of the backbone, the projector.

    #     Args:
    #         X (torch.Tensor): a batch of images in the tensor format.

    #     Returns:
    #         Dict[str, Any]:
    #             a dict containing the outputs of the parent
    #             and the projected features.
    #     """

    #     out = super().forward(X, *args, **kwargs)
    #     z = self.classifier(out["feats"])
    #     return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SupCon reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SupCon loss and classification loss.
        """

        targets = batch[-1]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats = out["feats"]

        # z = torch.cat([self.projector(f) for f in feats])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        # targets = targets.repeat(n_augs)
        assert len(feats) == 1 and n_augs == 1

        return class_loss
        # nce_loss = nn.functional.cross_entropy(z, targets)

        # self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        # return nce_loss + class_loss
