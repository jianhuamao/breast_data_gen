
DATA_FOLDER='./data'
DEVICES=1
TRAIN_MODEL='single'
CUDA_VISIBLE_DEVICES=$DEVICES python3 main.py \
    --name 'train_HorizontalFlip_and_Rotate_3channels'\
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_epochs 500 \
    --image_size 256 \
    --data_folder $DATA_FOLDER \
    --train_model $TRAIN_MODEL \
    --in_channels 6 \
    --start_epoch 20 \
    --lr 1e-5 \
    --pretrain_model_path './ckpt/train_HorizontalFlip_3channels_epoch_20.pth'\
    # --isDebug ture\
    # --nproc_per_node 4