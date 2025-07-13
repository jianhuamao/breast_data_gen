
DATA_FOLDER='./data'
DEVICES=0
TRAIN_MODEL='single'
CUDA_VISIBLE_DEVICES=$DEVICES python3 main.py \
    --train_batch_size 200 \
    --eval_batch_size 200 \
    --num_epochs 500 \
    --image_size 64 \
    --data_folder $DATA_FOLDER \
    --train_model $TRAIN_MODEL \
    --in_channels 6 \
    --start_epoch 0 \
    # --pretrain_model_path './ckp/epoch_199.pth'
    # --nproc_per_node 4