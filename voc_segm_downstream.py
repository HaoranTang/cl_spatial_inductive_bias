import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from PIL import Image
# NUM_CPU = os.cpu_count()
NUM_CPU = 2

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]
VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

class PascalVOCDataset(VOCSegmentation):
    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)

        return segmentation_mask.astype(np.uint8)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # img = Image.open(self.images[index]).convert("RGB")
        # target = Image.open(self.masks[index])
        mask = self._convert_to_segmentation_mask(mask)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.target_transform(mask)

        return F.resize(image, (224,224)), F.resize(mask, (224,224), interpolation=Image.NEAREST)

 
class VOC_SEG(Dataset):
    def __init__(self, root, width, height, train=True, transforms=None):
        # 图像统一剪切尺寸（width, height）
        self.width = width
        self.height = height
        # VOC数据集中对应的标签
        self.classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
        # 各种标签所对应的颜色
        self.colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]
        # 辅助变量
        self.fnum = 0
        if transforms is None:
            normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])
        # 像素值(RGB)与类别label(0,1,3...)一一对应
        self.cm2lbl = np.zeros(256**3)
        for i, cm in enumerate(self.colormap):
            self.cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
 
        
        if train:
            txt_fname = root+"/ImageSets/Segmentation/train.txt"
        else:
            txt_fname = root+"/ImageSets/Segmentation/val.txt"
        with open(txt_fname, 'r') as f:
            images = f.read().split()
        self.imgs = [os.path.join(root, "JPEGImages", item+".jpg") for item in images]
        self.labels = [os.path.join(root, "SegmentationClass", item+".png") for item in images]
        # self.imgs = self._filter(imgs)
        # self.labels = self._filter(labels)
        if train:
            print("训练集：加载了 " + str(len(self.imgs)) + " 张图片和标签")
        else:
            print("测试集：加载了 " + str(len(self.imgs)) + " 张图片和标签")
 
    def _crop(self, data, label):
        """
        切割函数，默认都是从图片的左上角开始切割。切割后的图片宽是width,高是height
        data和label都是Image对象
        """
        box = (0,0,self.width,self.height)
        data = data.crop(box)
        label = label.crop(box)
        return data, label
 
    def _image2label(self, im):
        data = np.array(im, dtype="int32")
        idx = (data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
        return np.array(self.cm2lbl[idx], dtype="int64")
        
    def _image_transforms(self, data, label):
        # data, label = self._crop(data,label)
        data = self.transforms(data.resize((self.width,self.height)))
        label = self._image2label(label.resize((self.width,self.height), Image.NEAREST))
        label = torch.from_numpy(label)
        return data, label
 
    def _filter(self, imgs): 
        img = []
        for im in imgs:
            if (Image.open(im).size[1] >= self.height and 
               Image.open(im).size[0] >= self.width):
                img.append(im)
            else:
                self.fnum  = self.fnum+1
        return img
 
    def __getitem__(self, index: int):
        img_path = self.imgs[index]
        label_path = self.labels[index]
        img = Image.open(img_path)
        label = Image.open(label_path).convert("RGB")
        img, label = self._image_transforms(img, label)
        return img, label
 
    def __len__(self) :
        return len(self.imgs)
 
 
# if __name__=="__main__":
#     root = "dataset/VOCdevkit/VOC2012"
#     height = 224
#     width = 224
#     voc_train = VOC_SEG(root, width, height, train=True)
#     voc_test = VOC_SEG(root, width, height, train=False)
 
#     # train_data = DataLoader(voc_train, batch_size=8, shuffle=True)
#     # valid_data = DataLoader(voc_test, batch_size=8)
#     for data, label in voc_train:
#         print(data.shape)
#         print(label.shape)
#         break
 

class SegmModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.classes = out_classes
        # dice loss for image segmentation
        self.dice_loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE)
        self.focal_loss_fn = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # image = batch["image"]
        image = batch[0]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # mask = batch["mask"]
        mask = batch[1]

        logits_mask = self.forward(image)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = 1.5*self.dice_loss_fn(logits_mask, mask) + self.focal_loss_fn(logits_mask, mask)

        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)
        # pred_mask = F.one_hot(pred_mask.long(), self.classes).permute(0,3,1,2)
        # mask = F.one_hot(mask.long(), self.classes).permute(0,3,1,2)


        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, mask, mode="multiclass", num_classes=self.classes)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

def compute_iou(predicted_mask, target_mask, class_idx):
    # print(f'checking value for class {class_idx}: {predicted_mask.unique()} and {target_mask.unique()}')
    intersection = (predicted_mask & target_mask).sum().item()
    union = (predicted_mask | target_mask).sum().item()
    iou = intersection / (union + 1e-10)  # Adding a small constant to avoid division by zero
    return iou

def compute_miou(predicted_masks, target_masks):
    # print(f'checking shape: {predicted_masks.shape} and {target_masks.shape}')
    batch_size, num_classes, _, _ = predicted_masks.shape
    ious = []

    for batch_idx in range(batch_size):
        iou_per_batch = []
        for class_idx in range(num_classes):
            predicted_mask = predicted_masks[batch_idx, class_idx]
            target_mask = target_masks[batch_idx, class_idx]
            iou = compute_iou(predicted_mask, target_mask, class_idx)
            iou_per_batch.append(iou)
        
        ious.append(iou_per_batch)
    
    ious = torch.tensor(ious)
    # print(ious)
    miou = torch.mean(ious)
    return miou

def main():
    # transform = transforms.Compose([
    #     transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    # ])

    # Create an instance of the VOCSegmentation dataset
    # CONFIGURE ROOT IF DOWNLOADED
    # train_dataset = VOCSegmentation(root='dataset', year='2012', image_set='train', download=False, transform=transform, target_transform=transform)
    # valid_dataset = VOCSegmentation(root='dataset', year='2012', image_set='val', download=False, transform=transform, target_transform=transform)
    # train_dataset = PascalVOCDataset(root='dataset', year='2012', image_set='train', download=False, transform=transform, target_transform=transform)
    # valid_dataset = PascalVOCDataset(root='dataset', year='2012', image_set='val', download=False, transform=transform, target_transform=transform)

    root = "dataset/VOCdevkit/VOC2012"
    height = 224
    width = 224
    train_dataset = VOC_SEG(root, width, height, train=True)
    valid_dataset = VOC_SEG(root, width, height, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2*NUM_CPU)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2*NUM_CPU)
    
    model = SegmModel("UNet", "resnet50", in_channels=3, out_classes=21)
    # model.model.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False)
    # CHANGE CKPT PATH
    ckpt_path = '/u/immortalco/bbhf/robust_imagenet/trained_models/byol-resnet50-imagenet-100ep/gam_0.2/byol-resnet50-imagenet-100ep-gam_0.2-ep=100.ckpt'
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
        # del state[k]
    model.model.encoder.load_state_dict(state, strict=False)
    # model.model.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=100,
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader
    )

    batch = next(iter(valid_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch[0])

    pr_masks = logits.softmax(dim=1).argmax(dim=1)

    count = 0
    for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
        count += 1
        print(gt_mask.shape, pr_mask.shape, gt_mask.unique())

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        # plt.savefig(f'img_{count}.png')
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        # plt.savefig(f'gt_{count}.png')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy()) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.savefig(f'pred_{count}.png')
        plt.axis("off")
    


if __name__ == "__main__":
    main()
