
DATA_FOLDER='./data'
DEVICES=0
TRAIN_MODEL='single'
CUDA_VISIBLE_DEVICES=$DEVICES python3 main.py \
    --name 'SD'\
    --model_name 'stabel_diffusion'\
    --device $DEVICES\
    --train_batch_size 10 \
    --eval_batch_size 10 \
    --num_epochs 800 \
    --image_size 256 \
    --data_folder $DATA_FOLDER \
    --train_model $TRAIN_MODEL \
    --in_channels 6 \
    --start_epoch 0 \
    --lr 1e-5 \
    --rank 16\
    --pretrain_model_path './my_model/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/stabel-diffusion'\
    # --isDebug ture\
    # --lora_path './ckpt/train_without_transform_epoch_500.pth'\
    # --isDebug ture\
    # --nproc_per_node 4