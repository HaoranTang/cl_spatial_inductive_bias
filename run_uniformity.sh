echo $0 ${1:-ori}
python3 main_uniformity.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --data_dir ./dataset \
    --train_dir cifar10/train \
    --val_dir cifar10/val \
    --max_epochs 20 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.3 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 20 \
    --pretrained_feature_extractor trained_models/mocov2plus-res18-cifar10/ori/mocov2plus-cifar10-ori-ep=200.ckpt \
    --name finetune-mocov2plus \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --csv \
    --save_checkpoint \
    --corrupt ${1:-ori} \