
DATA_FOLDER='/home/mjh_7515/diffusion/data_enhance/data'
DEVICES=4,5
TRAIN_MODEL='single'
CUDA_VISIBLE_DEVICES=$DEVICES python3 main.py \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 100 \
    --image_size 64 \
    --data_folder $DATA_FOLDER \
    --train_model $TRAIN_MODEL \
    --in_channels 3 \
    # --nproc_per_node 4