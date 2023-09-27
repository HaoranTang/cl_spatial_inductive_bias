ep=30
python3 main_pretrain.py \
    --dataset imagenetcls \
    --backbone resnet18 \
    --data_dir datasets/imagenet100bin \
    --max_epochs ${ep} \
    --gpus 0,1,2,3 \
    --accelerator ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.4 \
    --weight_decay 3e-5 \
    --batch_size 256 \
    --num_workers 7 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.55 \
    --solarization_prob 0.1 \
    --num_crops_per_aug 1 \
    --name sup-resnet18-in100-${ep}ep \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method sup \
    --csv \
    --checkpoint_frequency=50 \
    --corrupt ${1:-ori} \
    --knn_eval --knn_k=200

# --gaussian_prob 1.0 0.1 \
# --solarization_prob 0.0 0.2 \
# --gaussian_prob 0.0 \
# --saturation 0.4 \
# --crop_size 32 \
# --lars \  --exclude_bias_n_norm \
#     --grad_clip_lars \
#     --eta_lars 0.02 \
# --weight_decay 1e-5 \
# --momentum_classifier \