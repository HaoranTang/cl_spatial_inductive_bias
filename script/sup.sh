python3 main_pretrain.py \
    --dataset cifar10 \
    --backbone resnet50 \
    --data_dir $SCRATCH/datasets \
    --max_epochs 200 \
    --gpus 0,1,2,3,4,5,6,7,8 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.4 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 12 \
    --crop_size 32 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 \
    --name sup-$1 \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method sup \
    --csv \
    --checkpoint_frequency=50 \
    --corrupt ${2:-ori} \
    --knn_eval --knn_k=200


# --lars \
#     --grad_clip_lars \
#     --eta_lars 0.02 \