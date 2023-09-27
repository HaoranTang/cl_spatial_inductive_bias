echo $0 ${1:-ori}
rsync -r --progress /scratch/bbhf/imagenet2012Full /dev/shm
python3 main_pretrain.py \
    --dataset imagenetcls \
    --backbone vit_small \
    --data_dir /dev/shm/imagenet2012Full \
    --max_epochs 200 \
    --gpus 0,1,2,3,4,5,6,7 \
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
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 \
    --name mocov3-vit-in1k \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method mocov3 \
    --csv \
    --checkpoint_frequency=10 \
    --corrupt ${1:-ori} \
    # --knn_eval --knn_k=200
    # --lr 0.4 \
    # --classifier_lr 0.4 \
