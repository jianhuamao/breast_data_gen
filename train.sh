
DATA_FOLDER='./data'
DEVICES=0
TRAIN_MODEL='single'
CUDA_VISIBLE_DEVICES=$DEVICES python3 main.py \
    --name 'train_without_transform'\
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_epochs 500 \
    --image_size 256 \
    --data_folder $DATA_FOLDER \
    --train_model $TRAIN_MODEL \
    --in_channels 3 \
    --start_epoch 0 \
    --lr 1e-5 \
    # --pretrain_model_path './ckpt/epoch_1.pth'
    # --nproc_per_node 4