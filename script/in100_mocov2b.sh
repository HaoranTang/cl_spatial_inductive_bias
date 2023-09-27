ep=200
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
    --lr 0.3 \
    --classifier_lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --num_workers 6 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name mocov2b-resnet18-in100-${ep}ep \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 65536 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier \
    --csv \
    --checkpoint_frequency=50 \
    --corrupt ${1:-ori} \
    --knn_eval --knn_k=200 $2

# --auto_resume \
# --dali \
# --wandb \
# --resume_from_checkpoint=./trained_models/mocov2plus-resnet18-imagenet-100ep/ori/mocov2plus-resnet18-imagenet-100ep-ori-ep=75.ckpt
# --resume_from_checkpoint=./trained_models/_mocov2-resnet18-in100-100ep/ori/mocov2-resnet18-in100-100ep-ori-ep=75.ckpt
# --checkpoint_frequency=25 \
