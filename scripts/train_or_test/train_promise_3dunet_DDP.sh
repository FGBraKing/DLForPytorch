set -ex
echo "now is in $(pwd)"
#export MASTER_ADDR=localhost
#export MASTER_PORT=34567
export CUDA_VISIBLE_DEVICES="0,1,2"
cd /home/users/lf/CODE/PycharmProjects/DLForPytorch
python -m torch.distributed.launch --nproc_per_node 3 \
                train.py  \
                --name promise_unet_testDDP_45 \
                --dataset_name promise12 \
                --model_name unet3d \
                --seed 1008 \
                --gpu_ids 0,1,2 \
                --visible_gpu 0,1,2,3 \
                \
                --dataroot /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/datasets/promise12 \
                --phase 'train' \
                --gaussian_sigma '0.0,0.1'  \
                --crop_size '128,128,32' \
                --preprocess 'GaussianNoise_crop_rotate_centercrop_rot90_flip_bothscale' \
                --angle_spectrum  45   \
                --custom  \
                \
                --num_threads 1  \
                --batch_size  6  \
                \
                --input_nc  1  \
                --output_nc 1  \
                --conv_order  'crb'  \
                --init_channel_number  32 \
                \
                --init_type  'normal'  \
                --init_gain  0.02  \
                \
                --lr 1e-4 \
                --lr_policy linear  \
                --beta1 0.9  \
                --epoch_start 1 --epoch_retain 500  --epochs_decay 500 \
                \
                --logs_dir /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/logs \
                --checkpoints_dir  /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints  \
                --weight_path  None  \
                \
                --suffix ''  --DEBUG  \
                --DDP   \
                --dist_backend "nccl" --dist_url "tcp://172.21.141.4:30303" --world_size 3   \
                \
                --no_html  --tensorboard  --save_log \
                --display_freq 64  --print_freq 1   --save_epoch_freq 50  --save_iter_freq 500 \
                \
                --save_epoch_start 500   --save_iter_start 5000  \

                # --continue_train
                #                --verbose \

