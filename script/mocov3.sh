python3 main_pretrain.py \
    --dataset cifar10 \
    --backbone vit_small \
    --data_dir /home/haoran/Downloads/data \
    --max_epochs 200 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer lars \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 20 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name mocov3 \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method mocov3 \
    --proj_hidden_dim 4096 \
    --proj_output_dim 256 \
    --pred_hidden_dim 4096 \
    --temperature 0.3 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier \
    --csv \
    --checkpoint_frequency=20 \
    --corrupt ${1:-ori} \
    # --knn_eval --knn_k=200

    # --lr 0.3 \
    # --classifier_lr 0.3 \
    # --weight_decay 1e-4 \
    # --batch_size 256 \

    # --temperature 0.2 \