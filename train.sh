
DATA_FOLDER='./data'
DEVICES=0
TRAIN_MODEL='single'
CUDA_VISIBLE_DEVICES=$DEVICES python3 main.py \
    --train_batch_size 12 \
    --eval_batch_size 12 \
    --num_epochs 500 \
    --image_size 256 \
    --data_folder $DATA_FOLDER \
    --train_model $TRAIN_MODEL \
    --in_channels 3 \
    --start_epoch 0 \
    --lr 5e-5 \
    # --pretrain_model_path './ckpt/epoch_1.pth'
    # --nproc_per_node 4