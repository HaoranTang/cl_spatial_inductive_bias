echo $0 ${1:-ori}
python3 main_pretrain.py \
    --dataset cifar10 \
    --backbone resnet50 \
    --data_dir ./dataset \
    --max_epochs 200 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
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
    --name mocov2plus-cifar10 \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 32768 \
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