echo $0 ${1:-ori}
python3 main_pretrain.py \
    --dataset cifar10 \
    --backbone vit_small \
    --data_dir ./dataset \
    --max_epochs 50 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer adamw \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 3e-4 \
    --classifier_lr 3e-4 \
    --eta_lars 0.02 \
    --weight_decay 1e-5 \
    --batch_size 128 \
    --num_workers 20 \
    --crop_size 32 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 \
    --name sup-vit-cifar10 \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method sup \
    --csv \
    --checkpoint_frequency=10 \
    --corrupt ${1:-ori} \
    # --knn_eval --knn_k=200
    # --lr 0.4 \
    # --classifier_lr 0.4 \
