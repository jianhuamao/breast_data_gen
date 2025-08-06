
DATA_FOLDER='./data'
DEVICES=1
TRAIN_MODEL='single'
CUDA_VISIBLE_DEVICES=$DEVICES python3 main.py \
    --name 'HR_mean_and_std_0.5_3channels'\
    --device $DEVICES\
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --num_epochs 500 \
    --image_size 256 \
    --data_folder $DATA_FOLDER \
    --train_model $TRAIN_MODEL \
    --in_channels 6 \
    --start_epoch 0 \
    --lr 1e-5 \
    --isDebug ture
    # --pretrain_model_path './ckpt/train_HorizontalFlip_3channels_epoch_20.pth'\
    # --isDebug ture\
    # --nproc_per_node 4