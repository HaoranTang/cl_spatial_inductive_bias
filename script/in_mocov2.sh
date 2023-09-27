python3 main_pretrain.py \
    --dataset imagenetcls \
    --backbone resnet18 \
    --data_dir datasets/imagenet2012Full \
    --max_epochs 100 \
    --gpus 0,1,2,3 \
    --accelerator ddp \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.4 \
    --weight_decay 3e-5 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name mocov2plus-resnet18-imagenet-100ep \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 65536 \
    --temperature 0.1 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier \
    --auto_resume \
    --csv \
    --checkpoint_frequency=25 \
    --corrupt ${1:-ori} \
    --knn_eval --knn_k=200

# --sync_batchnorm \
# --dali \
# --wandb \
# --gpus 0,1,2,3 \